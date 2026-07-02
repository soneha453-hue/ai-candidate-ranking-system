import sys
import asyncio

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import nest_asyncio
nest_asyncio.apply()
import re

#**********************************imports*******************************************************************
from langgraph.graph import StateGraph,START,END
import tempfile
from psycopg_pool import AsyncConnectionPool
from typing import TypedDict ,Annotated,Any,Dict,Optional
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from  langchain_core.messages import BaseMessage,HumanMessage,SystemMessage,trim_messages
from langgraph.graph.message import add_messages
from langchain_tavily import TavilySearch
from langgraph.prebuilt import ToolNode,tools_condition
from langchain_core.tools import  tool
from langchain_core.runnables import RunnableConfig

import requests
import random
import ast
import operator
import threading
import os
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

load_dotenv()

DB_URL = os.getenv("DATABASE_URL")

# ----------------------------------------------------------------------------------
# Create a dedicated background event loop, running forever in its own thread,
# just for the Postgres checkpointer. This avoids "loop already running"
# conflicts with whatever event loop uvicorn/FastAPI is using to import this
# module or to serve requests later.
# ----------------------------------------------------------------------------------



import nest_asyncio
nest_asyncio.apply()

_loop = asyncio.new_event_loop()
_loop_thread = threading.Thread(target=_loop.run_forever, daemon=True)
_loop_thread.start()

def _run_async(coro):
    future = asyncio.run_coroutine_threadsafe(coro, _loop)
    return future.result(timeout=30)

async def _init_checkpointer():
    global checkpointer, _checkpointer_cm
    _checkpointer_cm = AsyncPostgresSaver.from_conn_string(DB_URL)
    checkpointer = await _checkpointer_cm.__aenter__()
    await checkpointer.setup()

_run_async(_init_checkpointer())


async def _ensure_connection():
    """
    Neon's free tier suspends the compute and drops idle connections.
    checkpointer.conn ab pool hai, isliye pool se ek connection borrow
    karke ping karte hain, single connection nahi.
    """
    global checkpointer, _pool
    try:
        async with checkpointer.conn.connection() as conn:
            await conn.execute("SELECT 1")
    except Exception as e:
        print(f"Connection dead, reconnecting... ({e})")
        try:
            await _pool.close()
        except Exception:
            pass
        _pool = AsyncConnectionPool(
            conninfo=DB_URL,
            max_size=5,
            kwargs={"autocommit": True, "keepalives": 1, "keepalives_idle": 30},
            open=False,
        )
        await _pool.open()
        checkpointer = AsyncPostgresSaver(_pool)
        await checkpointer.setup()

def run_async_with_reconnect(coro_factory):
    """
    Run a coroutine on the dedicated loop. Pings connection first, and if
    the query still fails mid-way (connection died in between), reconnects
    once and retries.
    """
    async def _wrapped():
        await _ensure_connection()
        try:
            return await coro_factory()
        except Exception as e:
            print(f"Query failed mid-way, forcing reconnect and retrying... ({e})")
            await _ensure_connection()
            return await coro_factory()

    return asyncio.run_coroutine_threadsafe(_wrapped(), _loop).result()
#*********************************************LLM*********************************************************

llm = ChatGroq(model='meta-llama/llama-4-scout-17b-16e-instruct')

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)
#******************************************pdf_retriever with chromadb*****************************************************
_THREAD_RETRIEVERS: Dict[str, list] = {}  
_THREAD_METADATA: Dict[str, list] = {}  


def _get_retriever(thread_id: Optional[str]):
    if thread_id and str(thread_id) in _THREAD_RETRIEVERS:
        retrievers = _THREAD_RETRIEVERS[str(thread_id)]
        return retrievers[-1] if retrievers else None  
    return None

import json

def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict:
    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        if not docs:
            raise ValueError("PDF load failed: no readable pages")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
        )

        chunks = splitter.split_documents(docs)

        if not chunks or len(chunks) == 0:
            raise ValueError("PDF se text extract nahi hua (chunks empty)")

        tid = str(thread_id)

        if tid not in _THREAD_RETRIEVERS:
            _THREAD_RETRIEVERS[tid] = []
            _THREAD_METADATA[tid] = []

        doc_index = len(_THREAD_RETRIEVERS[tid])

        # Batch embeddings taaki bade PDFs mein API limit hit na ho
        texts = [c.page_content for c in chunks]
        BATCH_SIZE = 50
        vector_store = None

        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i + BATCH_SIZE]
            if vector_store is None:
                vector_store = FAISS.from_texts(batch, embeddings)
            else:
                vector_store.add_texts(batch)

        retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}
        )

        _THREAD_RETRIEVERS[tid].append(retriever)

        meta = {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
            "index": doc_index
        }

        _THREAD_METADATA[tid].append(meta)

        try:
            save_path = f"faiss_store/{thread_id}/{doc_index}"
            os.makedirs(save_path, exist_ok=True)
            vector_store.save_local(save_path)
            with open(f"{save_path}/metadata.json", "w") as f:
                json.dump(meta, f)
        except Exception as e:
            print(f"FAISS save warning: {e}")

        return meta

    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass
 #**********************************************JSON ingestion (ADDITIVE ONLY)*********************************************

from langchain_core.documents import Document

def ingest_json(file_path: str, thread_id: str, filename: Optional[str] = None, batch_size: int = 200) -> dict:
    """
    Streams a .json or .jsonl file line-by-line / item-by-item so large files
    (like candidates.jsonl) don't blow up memory. Builds/extends a FAISS index
    for the given thread, same way ingest_pdf does.
    """
    if not os.path.exists(file_path):
        raise ValueError("No file found for ingestion.")

    tid = str(thread_id)

    if tid not in _THREAD_RETRIEVERS:
        _THREAD_RETRIEVERS[tid] = []
        _THREAD_METADATA[tid] = []

    doc_index = len(_THREAD_RETRIEVERS[tid])

    is_jsonl = file_path.endswith(".jsonl")

    vector_store = None
    total_items = 0
    batch_docs = []

    def _flush(batch):
        nonlocal vector_store
        if not batch:
            return
        if vector_store is None:
            vector_store = FAISS.from_documents(batch, embeddings)
        else:
            vector_store.add_documents(batch)

    try:
        if is_jsonl:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    text = json.dumps(item, ensure_ascii=False)
                    batch_docs.append(Document(page_content=text, metadata={"source": filename}))
                    total_items += 1
                    if len(batch_docs) >= batch_size:
                        _flush(batch_docs)
                        batch_docs = []
            _flush(batch_docs)
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            items = data if isinstance(data, list) else [data]
            for item in items:
                text = json.dumps(item, ensure_ascii=False)
                batch_docs.append(Document(page_content=text, metadata={"source": filename}))
                total_items += 1
                if len(batch_docs) >= batch_size:
                    _flush(batch_docs)
                    batch_docs = []
            _flush(batch_docs)

        if vector_store is None:
            raise ValueError("No items found to index in the JSON file.")

        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
        _THREAD_RETRIEVERS[tid].append(retriever)

        meta = {
            "filename": filename or os.path.basename(file_path),
            "documents": total_items,
            "chunks": total_items,
            "index": doc_index
        }
        _THREAD_METADATA[tid].append(meta)

        try:
            save_path = f"faiss_store/{thread_id}/{doc_index}"
            os.makedirs(save_path, exist_ok=True)
            vector_store.save_local(save_path)
            with open(f"{save_path}/metadata.json", "w") as f:
                json.dump(meta, f)
        except Exception as e:
            print(f"FAISS save warning: {e}")

        return meta

    finally:
        pass  # temp file cleanup handled by caller       
#**********************************************system prompt*********************************************
system_prompt = r"""You are a smart, helpful AI assistant with access to powerful tools. Always use the right tool for the right task — never make up answers when a tool can fetch accurate information.

## Who You Are
You are an expert assistant that can:
- Search the internet for latest news and information
- Check real-time weather for any city
- Fetch live stock prices
- Perform mathematical calculations
- Search through uploaded documents and PDFs
- Find relevant job listings based on user query or uploaded resume
- Rank and compare multiple candidate resumes against a job description

## Tool Usage Rules
ALWAYS use the appropriate tool — never guess or make up answers:
- User asks about weather → get_weather_info tool
- User asks about stocks/share price → get_stockprice tool
- User asks math/calculation → calculator tool
- User asks about news/current events/general knowledge → search tool
- User asks about uploaded PDF/document → rag_tool
- User asks about jobs/careers → search_jobs tool
- User wants to rank/compare PDF resumes → rank_candidates tool
- User wants to rank/compare JSON/JSONL candidate data → rank_json_candidates_v2 tool (preferred over rank_json_candidates — it includes behavioral signals and skill verification)


Never say "I don't have access" or "I cannot retrieve" — you have tools, use them.
Never reveal tool names to the user unless they specifically ask.

## Document & PDF Behavior
When a user uploads a PDF or document:
- Warmly acknowledge the upload
- Inform them they can ask questions about it
- If it looks like a resume, also offer job recommendations based on it
- For ANY question about the document → ALWAYS use rag_tool, no exceptions
- Never answer document questions from memory — always retrieve from the document
- NEVER say "I don't have the document" or "please upload again" — if a PDF was uploaded in this session, it is available. Just call rag_tool immediately.
- Do NOT ask the user to re-upload — trust that the document is indexed and use rag_tool

## Candidate Ranking Behavior
When candidates are uploaded and user wants to rank or compare them against a job description:
- If candidates were uploaded as separate PDF resumes → use rank_candidates tool
- If candidates were uploaded as JSON/JSONL data → use rank_json_candidates_v2 tool (includes semantic match, keyword match, availability/behavioral signals, and skill verification flags)
- If user has not provided a job description, ask for it first
- ALWAYS present results in a markdown table with columns: Rank | Candidate ID | Overall Score | Semantic Match | Keyword Match | Availability/Engagement | Skill Verification | Flags
- Never change the output format — always use table format
- Never rank from memory — always use the tool

### Duplicate Candidate Handling
- Before presenting the final ranked table, check if the same Candidate ID appears more than once in the tool's output.
- If a Candidate ID is duplicated, keep ONLY the entry with the higher Overall Score and discard the other(s) — never show the same Candidate ID twice in the final table.
- After removing duplicates, re-number the ranks sequentially (1, 2, 3...) with no gaps.
- Do this silently — do not mention to the user that duplicates were found or removed unless they specifically ask.

## Job Search Behavior
When user asks about jobs or career opportunities:
1. If resume is uploaded (pdf_context will say so):
   - First use rag_tool to extract skills, experience, and job preferences from resume
   - Then use search_jobs tool with those extracted skills and location
   - Do NOT ask user to upload resume again if pdf_context confirms it is already uploaded
2. If no resume uploaded:
   - Ask user: what kind of job? preferred location?
   - Then use search_jobs tool accordingly
3. Always return results with: job title, company, location, salary (if available), apply link
4. Suggest 3-5 most relevant jobs only — no irrelevant listings

## Communication Style
- Answer first, explain later if needed
- Match user's language — Hindi, English, or Hinglish
- Match user's language — Hindi, English, or Hinglish
- Keep responses concise and scannable
- Use markdown formatting properly:
  - **bold** for important terms
  - \`code\` for technical terms
  - Tables for comparisons
  - Numbered steps for processes
  - Bullet points for lists
- One-line question → one-line answer
- Never write walls of text

## What You Do NOT Know
If user asks something completely unrelated to your capabilities (random trivia, personal opinions, inappropriate topics):
- Politely say: "I don't have information about this, but I can help you with weather, stocks, job search, document questions, or general web search!"
- Never make up answers
- Never hallucinate facts

## What To Avoid
- Do not add unnecessary headings like "Summary" or "Conclusion"
- Do not repeat the user's question
- Do not dump raw search results
- Do not show URLs unless user asks
- Do not use excessive emojis
- Do not reveal which tool you are using

Think like a knowledgeable human expert — smart, direct, and always helpful.
"""

#**********************************************Per-candidate ranking (ADDITIVE ONLY)*********************************************

_THREAD_CANDIDATES: Dict[str, list] = {}

def _tokenize(text: str) -> set:
    return set(re.findall(r"[a-z0-9]+", text.lower()))

def ingest_candidates_json(file_path: str, thread_id: str, filename: Optional[str] = None) -> dict:
    """
    Stores each JSON/JSONL record as its OWN candidate entry (no chunking),
    so rank_json_candidates can score them individually.
    """
    tid = str(thread_id)
    if tid not in _THREAD_CANDIDATES:
        _THREAD_CANDIDATES[tid] = []

    is_jsonl = file_path.endswith(".jsonl")
    count = 0

    if is_jsonl:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                _THREAD_CANDIDATES[tid].append(item)
                count += 1
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        items = data if isinstance(data, list) else [data]
        for item in items:
            _THREAD_CANDIDATES[tid].append(item)
            count += 1

    return {"filename": filename or os.path.basename(file_path), "candidates_loaded": count}


@tool
def rank_json_candidates(job_description: str, config: RunnableConfig) -> str:
    """
    Rank individually-loaded JSON candidates (from ingest_candidates_json) against a job description.
    Use this instead of rank_candidates when candidates were uploaded as JSON/JSONL, not as separate PDF resumes.
    """
    from sklearn.metrics.pairwise import cosine_similarity

    thread_id = config.get("configurable", {}).get("thread_id")
    candidates = _THREAD_CANDIDATES.get(str(thread_id), [])

    if not candidates:
        return "No JSON candidates loaded for this chat. Please upload a candidates JSON/JSONL file first."

    jd_embedding = embeddings.embed_query(job_description)
    jd_keywords = _tokenize(job_description)

    results = []
    for c in candidates:
        cand_text = json.dumps(c, ensure_ascii=False).lower()
        cand_words = _tokenize(cand_text)

        cand_embedding = embeddings.embed_query(cand_text[:2000])
        semantic_score = round(
            float(cosine_similarity([jd_embedding], [cand_embedding])[0][0]) * 50, 2
        )

        matched_keywords = jd_keywords & cand_words
        skill_score = round(len(matched_keywords) / max(len(jd_keywords), 1) * 30, 2)

        final_score = round(semantic_score + skill_score, 2)

        results.append({
            "candidate_id": c.get("candidate_id", "unknown"),
            "semantic_score": semantic_score,
            "skill_score": skill_score,
            "final_score": final_score,
            "matched_keywords": list(matched_keywords)[:10]
        })

    results.sort(key=lambda x: x["final_score"], reverse=True)

    output = "## Candidate Ranking Results (JSON)\n\n"
    for i, r in enumerate(results[:20]):  # top 20 dikhao
        output += f"**Rank {i+1} — {r['candidate_id']}**\n"
        output += f"- Overall Score: {r['final_score']}/80\n"
        output += f"- Semantic Match: {r['semantic_score']}/50\n"
        output += f"- Skill Match: {r['skill_score']}/30\n"
        output += f"- Matched Keywords: {', '.join(r['matched_keywords'])}\n\n"

    return output

#**********************************************Enhanced ranking with behavioral signals (ADDITIVE ONLY)*********************************************

@tool
def rank_json_candidates_v2(job_description: str, config: RunnableConfig) -> str:
    """
    Rank JSON candidates against a job description using semantic match, keyword match,
    AND behavioral/availability signals (redrob_signals) — not just keyword matching.
    Flags suspicious profiles where self-reported skills don't match verified assessment scores.
    Use this as the primary ranking tool for hackathon submission quality.
    """
    from sklearn.metrics.pairwise import cosine_similarity
    from datetime import datetime

    thread_id = config.get("configurable", {}).get("thread_id")
    candidates = _THREAD_CANDIDATES.get(str(thread_id), [])

    if not candidates:
        return "No JSON candidates loaded for this chat. Please upload a candidates JSON/JSONL file first."

    jd_embedding = embeddings.embed_query(job_description)
    jd_keywords = _tokenize(job_description)          # 🔥 FIXED

    results = []
    for c in candidates:
        cand_text = json.dumps(c, ensure_ascii=False).lower()
        cand_words = _tokenize(cand_text)              # 🔥 FIXED

        # 1. Semantic match (0-40)
        cand_embedding = embeddings.embed_query(cand_text[:2000])
        semantic_score = round(
            float(cosine_similarity([jd_embedding], [cand_embedding])[0][0]) * 40, 2
        )

        # 2. Keyword match (0-20)
        matched_keywords = jd_keywords & cand_words
        skill_score = round(len(matched_keywords) / max(len(jd_keywords), 1) * 20, 2)

        # 3. Behavioral/availability signals (0-25)
        signals = c.get("redrob_signals", {})
        behavior_score = 0
        flags = []

        response_rate = signals.get("recruiter_response_rate", 0)
        behavior_score += response_rate * 10

        if not signals.get("open_to_work_flag", False):
            flags.append("not currently open to work")
        else:
            behavior_score += 5

        last_active = signals.get("last_active_date")
        if last_active:
            try:
                days_inactive = (datetime.now() - datetime.strptime(last_active, "%Y-%m-%d")).days
                if days_inactive > 180:
                    flags.append(f"inactive for {days_inactive} days")
                else:
                    behavior_score += max(0, 10 - (days_inactive / 30))
            except Exception:
                pass

        behavior_score = round(min(behavior_score, 25), 2)

        # 4. Skill verification check (0-15)
        skill_verify_score = 15
        assessment_scores = signals.get("skill_assessment_scores", {})
        skills_listed = c.get("skills", [])

        for skill in skills_listed:
            name = skill.get("name")
            proficiency = skill.get("proficiency", "")
            if name in assessment_scores:
                assessed = assessment_scores[name]
                if proficiency == "advanced" and assessed < 50:
                    flags.append(f"'{name}' claimed advanced but assessment score only {assessed}")
                    skill_verify_score -= 5

        skill_verify_score = round(max(skill_verify_score, 0), 2)

        final_score = round(semantic_score + skill_score + behavior_score + skill_verify_score, 2)

        results.append({
            "candidate_id": c.get("candidate_id", "unknown"),
            "semantic_score": semantic_score,
            "skill_score": skill_score,
            "behavior_score": behavior_score,
            "skill_verify_score": skill_verify_score,
            "final_score": final_score,
            "matched_keywords": list(matched_keywords)[:8],
            "flags": flags
        })

    results.sort(key=lambda x: x["final_score"], reverse=True)

    output = "## Candidate Ranking Results (Semantic + Behavioral + Skill Verification)\n\n"
    for i, r in enumerate(results[:20]):
        output += f"**Rank {i+1} — {r['candidate_id']}**\n"
        output += f"- Overall Score: {r['final_score']}/100\n"
        output += f"- Semantic Match: {r['semantic_score']}/40\n"
        output += f"- Keyword Match: {r['skill_score']}/20\n"
        output += f"- Availability/Engagement: {r['behavior_score']}/25\n"
        output += f"- Skill Verification: {r['skill_verify_score']}/15\n"
        if r['flags']:
            output += f"- ⚠️ Flags: {'; '.join(r['flags'])}\n"
        output += f"- Matched Keywords: {', '.join(r['matched_keywords'])}\n\n"

    return output
#************************************************Tools******************************************************


tavily_tool = TavilySearch(max_results=3)

@tool
def search(query: str) -> str:
    """Search for current information, news, and general knowledge."""
    results = tavily_tool.invoke(query)
    
    output_parts = []
    if results.get('answer'):
        output_parts.append(results['answer'])
    
    result_list = results.get('results', [])
    if not result_list:
        return "No relevant search results found."
    
    for r in result_list[:3]:
        content = r.get('content', '').strip()
        if content:
            output_parts.append(f"Source: {r.get('title','')}\n{content}")
    
    if not output_parts:
        return "No relevant search results found."
    
    return "\n\n".join(output_parts)

OPERATORS={
    ast.Add : operator.add,
    ast.Sub : operator.sub,
    ast.Mult: operator.mul,
    ast.Div : operator.truediv,
    ast.Mod :operator.mod,
    ast.Pow :operator.pow,
    ast.USub :operator.neg
}

def sava_eval(node):
    if isinstance(node,ast.Constant):
        return node.value
    elif isinstance(node,ast.BinOp):
        left=sava_eval(node.left)
        right=sava_eval(node.right)
        op=OPERATORS[type(node.op)]
        return op(left,right)
    elif isinstance(node,ast.UnaryOp):
        return  OPERATORS[type(node.op)](sava_eval(node.operand))
    else:
        raise ValueError("Unsupported operation")
    

@tool
def calculator(expression:str)->str:
    """ Use this tool when user asks any mathematical calculation.
    Input should be a valid math expression string.
    Examples: '2 + 3', '10 * 5', '(100 - 20) / 4', '2 ** 8'
    Supports: +, -, *, /, **, %
    """

    try:
        tree=ast.parse(expression,mode='eval')
        result=sava_eval(tree.body)

        return str(result)
    except Exception as e:
        return f" error {e}"
    

@tool
def get_stockprice(symbol:str)->str:
    """Get current stock price for a given symbol.
    Use for any stock price related questions.
    Example symbols: 'AAPL' for Apple, 'RELIANCE.NS' for Reliance India, 'TCS.NS' for TCS
    """
    url = f"https://finnhub.io/api/v1/quote?symbol={symbol}&token={os.getenv('FINNHUB_API_KEY')}"
    response =requests.get(url)
    data= response.json()
    return  f"Current price: {data['c']}, High: {data['h']}, Low: {data['l']}"



@tool 
def get_weather_info(city: str):
    """Use this tool to get current weather information for any city.
    Use when user asks about weather, temperature, humidity, or climate of a place.
    Input should be a city name string.
    Examples: 'London', 'Mumbai', 'New York', 'Delhi,UP'
    Returns current temperature, weather condition, humidity and wind speed.
    """
    url = f'http://api.openweathermap.org/data/2.5/weather?q={city}&appid={os.getenv("OPENWEATHER_API_KEY")}&units=metric'
    response = requests.get(url)
    data = response.json()

    if str(data.get('cod')) != '200':
        return f"Weather fetch failed: {data.get('message', 'unknown error')}"

    return (
        f"City: {city}, "
        f"Temp: {data['main']['temp']}°C, "
        f"Condition: {data['weather'][0]['description']}, "
        f"Humidity: {data['main']['humidity']}%, "
        f"Wind: {data['wind']['speed']} m/s"
    )




from langchain_core.runnables import RunnableConfig

@tool
def rag_tool(query: str, config: RunnableConfig) -> str:
    """Retrieve relevant information from uploaded PDFs for this chat thread."""
    thread_id = config.get("configurable", {}).get("thread_id")
    retrievers = _THREAD_RETRIEVERS.get(str(thread_id), [])
    
    if not retrievers:
        return "No document indexed for this chat. Please upload a PDF first."

    all_context = []
    metas = _THREAD_METADATA.get(str(thread_id), [])
    
    for i, retriever in enumerate(retrievers):
        result = retriever.invoke(query)
        context = [doc.page_content for doc in result]
        filename = metas[i].get("filename", f"document_{i}") if i < len(metas) else f"document_{i}"
        if context:
            all_context.append(f"Source: {filename}\n" + "\n".join(context))

    if not all_context:
        return "No relevant information found in the documents."

    return "\n\n---\n\n".join(all_context)


@tool
def rank_candidates(job_description: str, config: RunnableConfig) -> str:
    """
    Rank all uploaded resumes against a job description.
    Use when user wants to rank, compare, or shortlist candidates.
    Input: job description text.
    """
    import re
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    thread_id = config.get("configurable", {}).get("thread_id")
    retrievers = _THREAD_RETRIEVERS.get(str(thread_id), [])
    metas = _THREAD_METADATA.get(str(thread_id), [])

    if not retrievers:
        return "No resumes uploaded. Please upload candidate resumes first."
    if len(retrievers) == 1:
        return "Only one resume uploaded. Upload multiple resumes to rank candidates."

    # JD embed karo
    jd_embedding = embeddings.embed_query(job_description)
    jd_keywords = set(job_description.lower().split())

    results = []

    for i, retriever in enumerate(retrievers):
        filename = metas[i].get("filename", f"Candidate {i+1}") if i < len(metas) else f"Candidate {i+1}"

        docs = retriever.invoke(job_description)
        resume_text = " ".join([doc.page_content for doc in docs]).lower()
        resume_words = set(resume_text.split())

        # 1. Semantic similarity score (0-50)
        resume_embedding = embeddings.embed_query(resume_text[:2000])
        semantic_score = round(
            float(cosine_similarity([jd_embedding], [resume_embedding])[0][0]) * 50, 2
        )

        # 2. Skill match score (0-30)
        matched_keywords = jd_keywords & resume_words
        skill_score = round(len(matched_keywords) / max(len(jd_keywords), 1) * 30, 2)

        # 3. Experience score (0-20)
        years = re.findall(r'(\d+)\+?\s*year', resume_text)
        exp_score = min(int(years[0]) * 4, 20) if years else 2

        final_score = round(semantic_score + skill_score + exp_score, 2)

        results.append({
            "filename": filename,
            "semantic_score": semantic_score,
            "skill_score": skill_score,
            "exp_score": exp_score,
            "final_score": final_score,
            "matched_keywords": list(matched_keywords)[:10]
        })

    results.sort(key=lambda x: x["final_score"], reverse=True)

    output = "## Candidate Ranking Results\n\n"
    for i, r in enumerate(results):
        output += f"**Rank {i+1} — {r['filename']}**\n"
        output += f"- Overall Score: {r['final_score']}/100\n"
        output += f"- Semantic Match: {r['semantic_score']}/50\n"
        output += f"- Skill Match: {r['skill_score']}/30\n"
        output += f"- Experience: {r['exp_score']}/20\n"
        output += f"- Matched Keywords: {', '.join(r['matched_keywords'])}\n\n"

    return output


@tool
def search_jobs(query:str,location:str ="India")->str:
    """ Search for job listings by keyword and location.
    Use when user asks about jobs, careers, or employment opportunities.
    Input: job title or skill (e.g. 'Python developer', 'AI engineer')"""

    app_id = os.getenv("ADZUNA_APP_ID")
    app_key = os.getenv("ADZUNA_APP_KEY")
    url = f"https://api.adzuna.com/v1/api/jobs/in/search/1"
    param={
        "app_id":app_id,
        "app_key":app_key,
        "what":query,
        "where":location,
        "results_per_page":5,
        "content-type": "application/json"        
    }
    response=requests.get(url,params=param)
    data=response.json()

    jobs=[]
    for job in data.get("results", []):
        jobs.append({
            "title": job.get("title"),
            "company": job.get("company", {}).get("display_name"),
            "location": job.get("location", {}).get("display_name"),
            "salary": job.get("salary_min"),
            "url": job.get("redirect_url")
        })
    
    return str(jobs)
#**************************************tools Function with llms******************************************************

tools=[search,get_stockprice,get_weather_info,calculator,rag_tool,search_jobs,rank_candidates,rank_json_candidates,rank_json_candidates_v2 ]
 


llm_with_tools=llm.bind_tools(tools)
#*********************************************************State***********************************************
class chatState(TypedDict):
    messages:Annotated[list[BaseMessage],add_messages]

#***************************************************chat node*************************************************

def chat_node(state: chatState, config: RunnableConfig):
    messages = state['messages']

    thread_id = config.get("configurable", {}).get("thread_id", "unknown")

    has_pdf = str(thread_id) in _THREAD_RETRIEVERS and len(_THREAD_RETRIEVERS[str(thread_id)]) > 0
    pdf_context = ""
    if has_pdf:
        metas = _THREAD_METADATA.get(str(thread_id), [])
        filenames = [m.get("filename", "document") for m in metas]
        count = len(filenames)
        if count == 1:
            pdf_context = f"\n\nIMPORTANT: The user has uploaded {count} PDF file: {', '.join(filenames)}. For ANY question about this document, ALWAYS use rag_tool immediately. Do NOT say the document is missing or ask to re-upload."
        else:
            pdf_context = f"\n\nIMPORTANT: The user has uploaded {count} PDF files: {', '.join(filenames)}. For ANY question about these documents use rag_tool. If user wants to rank or compare candidates, ALWAYS use rank_candidates tool with the job description."

    final_system = system_prompt + pdf_context

    trim_msg = trim_messages(
        messages,
        max_tokens=200,
        strategy='last',
        token_counter=len,
        include_system=False,
        allow_partial=False
    )

    final_message = [SystemMessage(content=final_system)] + trim_msg
    response = llm_with_tools.invoke(final_message)
    return {'messages': [response]}

tool_node=ToolNode(tools,handle_tool_errors=True)

#****************************************************graph and edges*********************************************

graph=StateGraph(chatState)

graph.add_node('chat_node',chat_node)
graph.add_node('tools',tool_node)

graph.add_edge(START,'chat_node')
graph.add_conditional_edges('chat_node',tools_condition)
graph.add_edge('tools','chat_node')
graph.add_edge('chat_node',END)

chatbot=graph.compile(checkpointer=checkpointer)

#****************************************************reload-block*****************************************************************


import glob
for _thread_path in glob.glob("faiss_store/*"):
    _tid = os.path.basename(_thread_path)
    if not os.path.isdir(_thread_path):
        continue

    _THREAD_RETRIEVERS[_tid] = []
    _THREAD_METADATA[_tid] = []

    for _doc_path in sorted(glob.glob(f"{_thread_path}/*")):
        if not os.path.isdir(_doc_path):
            continue
        try:
            _vs = FAISS.load_local(_doc_path, embeddings, allow_dangerous_deserialization=True)
            _THREAD_RETRIEVERS[_tid].append(
                _vs.as_retriever(search_type="similarity", search_kwargs={"k": 4})
            )
            meta_path = os.path.join(_doc_path, "metadata.json")
            if os.path.exists(meta_path):
                with open(meta_path, "r") as f:
                    _THREAD_METADATA[_tid].append(json.load(f))
            else:
                _THREAD_METADATA[_tid].append({"filename": "uploaded document"})
            print(f"Reloaded FAISS: {_tid}/{os.path.basename(_doc_path)}")
        except Exception as _e:
            print(f"Could not reload {_doc_path}: {_e}")
#**********************************************Dynamic thread****************************************************

async def retriev_thread():
    async def _collect():
        all_thread = set()
        async for cp in checkpointer.alist(None):
            try:
                tid = cp.config['configurable']['thread_id']
                all_thread.add(tid)
            except (KeyError, TypeError):
                continue
        return list(all_thread)

    async def _wrapped():
        await _ensure_connection()
        return await _collect()

    return await asyncio.wrap_future(
        asyncio.run_coroutine_threadsafe(_wrapped(), _loop)
    )
#***************************************************delete thread_function****************************************

async def delete_thread(thread_id):
    async def _wrapped():
        await _ensure_connection()
        await checkpointer.adelete_thread(thread_id)

    await asyncio.wrap_future(
        asyncio.run_coroutine_threadsafe(_wrapped(), _loop)
    )

def close_checkpointer():
    """Call this on FastAPI shutdown to release the Postgres connection cleanly."""
    try:
        _run_async(_checkpointer_cm.__aexit__(None, None, None))
    except Exception:
        pass
# chatbot



async def _init_checkpointer():
    global checkpointer, _pool
    _pool = AsyncConnectionPool(
        conninfo=DB_URL,
        max_size=5,
        kwargs={"autocommit": True, "keepalives": 1, "keepalives_idle": 30},
        open=False,
        check=AsyncConnectionPool.check_connection,   # ye naya add hua
    )
    await _pool.open()
    checkpointer = AsyncPostgresSaver(_pool)
    await checkpointer.setup()

_run_async(_init_checkpointer())

#**********************************************Safe thread retrieval with retry (ADDITIVE ONLY)*********************************************

import psycopg

async def retriev_thread_safe():
    """
    Same as retriev_thread, but if the connection died mid-query
    (common with Neon's idle drops), it force-reconnects and retries once,
    instead of relying only on the proactive .closed check.
    """
    global checkpointer, _checkpointer_cm

    async def _collect():
        all_thread = set()
        async for cp in checkpointer.alist(None):
            try:
                tid = cp.config['configurable']['thread_id']
                all_thread.add(tid)
            except (KeyError, TypeError):
                continue
        return list(all_thread)

    async def _wrapped():
        global checkpointer, _checkpointer_cm
        try:
            return await _collect()
        except psycopg.OperationalError:
            print("Connection dead mid-query — force reconnecting...")
            try:
                await _checkpointer_cm.__aexit__(None, None, None)
            except Exception:
                pass
            _checkpointer_cm = AsyncPostgresSaver.from_conn_string(DB_URL)
            checkpointer = await _checkpointer_cm.__aenter__()
            return await _collect()

    return await asyncio.wrap_future(
        asyncio.run_coroutine_threadsafe(_wrapped(), _loop)
    )

# config={'configurable':{'thread_id':'threa_id-4'}}



# out=chatbot.invoke({'messages':[HumanMessage(content='what stock price of apple ')]},config=config)
# print(out['messages'][-1].content)