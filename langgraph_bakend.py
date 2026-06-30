import sys
import asyncio

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import nest_asyncio
nest_asyncio.apply()


#**********************************imports*******************************************************************
from langgraph.graph import StateGraph,START,END
import tempfile

from typing import TypedDict ,Annotated,Any,Dict,Optional
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import  Chroma
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
    If the checkpointer's connection is closed, reconnect it transparently.
    """
    global checkpointer, _checkpointer_cm
    try:
        if checkpointer.conn.closed:
            print("Checkpointer connection was closed — reconnecting...")
            try:
                await _checkpointer_cm.__aexit__(None, None, None)
            except Exception:
                pass
            _checkpointer_cm = AsyncPostgresSaver.from_conn_string(DB_URL)
            checkpointer = await _checkpointer_cm.__aenter__()
    except Exception as e:
        print(f"Reconnect check failed: {e}")


def run_async_with_reconnect(coro_factory):
    """
    Run a coroutine on the dedicated loop, retrying once after reconnecting
    if the Postgres connection turned out to be closed.
    """
    async def _wrapped():
        await _ensure_connection()
        return await coro_factory()

    return asyncio.run_coroutine_threadsafe(_wrapped(), _loop).result()
#*********************************************LLM*********************************************************

llm = ChatGroq(model='openai/gpt-oss-120b')

embeddings = HuggingFaceEmbeddings(
    model_name='BAAI/bge-base-en-v1.5'
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

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(docs)

        tid = str(thread_id)

        if tid not in _THREAD_RETRIEVERS:
            _THREAD_RETRIEVERS[tid] = []
            _THREAD_METADATA[tid] = []

        doc_index = len(_THREAD_RETRIEVERS[tid])
        
        
        save_path = f"chroma_store/{thread_id}/{doc_index}"
        os.makedirs(save_path, exist_ok=True)
        vector_store = Chroma.from_documents(
            chunks, embeddings, persist_directory=save_path
        )
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
            with open(f"{save_path}/metadata.json", "w") as f:
                json.dump(meta, f)
        except Exception as e:
            print(f"Metadata save warning: {e}")

        return meta

    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass
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
- User wants to rank/compare resumes → rank_candidates tool

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
When multiple resumes are uploaded and user wants to rank or compare:
- ALWAYS use rank_candidates tool with the job description provided by user
- If user has not provided a job description, ask for it first
- Return results clearly showing rank, score, and reason
- Never rank from memory — always use the tool

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

tools=[search,get_stockprice,get_weather_info,calculator,rag_tool,search_jobs,rank_candidates]
 


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
for _thread_path in glob.glob("chroma_store/*"):
    _tid = os.path.basename(_thread_path)
    if not os.path.isdir(_thread_path):
        continue

    _THREAD_RETRIEVERS[_tid] = []
    _THREAD_METADATA[_tid] = []

    for _doc_path in sorted(glob.glob(f"{_thread_path}/*")):
        if not os.path.isdir(_doc_path):
            continue
        try:
            _vs = Chroma(
                persist_directory=_doc_path,
                embedding_function=embeddings
            )
            _THREAD_RETRIEVERS[_tid].append(
                _vs.as_retriever(search_type="similarity", search_kwargs={"k": 4})
            )
            meta_path = os.path.join(_doc_path, "metadata.json")
            if os.path.exists(meta_path):
                with open(meta_path, "r") as f:
                    _THREAD_METADATA[_tid].append(json.load(f))
            else:
                _THREAD_METADATA[_tid].append({"filename": "uploaded document"})
            print(f"Reloaded Chroma: {_tid}/{os.path.basename(_doc_path)}")
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
# config={'configurable':{'thread_id':'threa_id-4'}}



# out=chatbot.invoke({'messages':[HumanMessage(content='what stock price of apple ')]},config=config)
# print(out['messages'][-1].content)