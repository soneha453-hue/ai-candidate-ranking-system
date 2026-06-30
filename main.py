
"""
FastAPI backend for the RAG multi-tool LangGraph chatbot.
 
Endpoints:
- POST /chat/stream      -> Stream chatbot response token-by-token (Server-Sent Events)
- POST /upload-pdf       -> Upload & ingest a PDF for a given thread (RAG)
- GET  /threads          -> List all existing thread ids
- DELETE /threads/{id}   -> Delete a thread and its checkpoint history
- GET  /health           -> Simple health check
 
Run with:
    uvicorn app:app --reload --port 8000
"""
 
import json
import uuid
import asyncio
from contextlib import asynccontextmanager
from typing import Optional
from functools import partial
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
 
from langchain_core.messages import AIMessageChunk, HumanMessage,AIMessage


 
from langgraph_bakend import (
    chatbot,
    ingest_pdf,
    retriev_thread,
    delete_thread,
    checkpointer,
    _loop,
    run_async_with_reconnect,  
)
 
 
# ----------------------------------------------------------------------------------
# Lifespan: close the Postgres checkpointer connection cleanly on shutdown
# ----------------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
 
 
app = FastAPI(title="RAG Multi-Tool Chatbot API", lifespan=lifespan)
 
# Allow your frontend (React/Streamlit/etc.) to call this API.
# Replace "*" with your actual frontend origin in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
 
 
# ----------------------------------------------------------------------------------
# Request/response schemas
# ----------------------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str
    thread_id: Optional[str] = None  # if not given, a new thread is created
 
 
class ThreadDeleteResponse(BaseModel):
    thread_id: str
    deleted: bool
 
 
# ----------------------------------------------------------------------------------
# Health check
# ----------------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}
 
 
# ----------------------------------------------------------------------------------
# Chat — streaming (token-by-token via Server-Sent Events)
# ----------------------------------------------------------------------------------
@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    thread_id = request.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    async def event_generator():
        yield f"data: {json.dumps({'thread_id': thread_id})}\n\n"

        try:
            loop = asyncio.get_event_loop()

            def run_chatbot():
                return run_async_with_reconnect(
                    lambda: chatbot.ainvoke(
                        {"messages": [HumanMessage(content=request.message)]},
                        config=config
                    )
                )

            result = await loop.run_in_executor(None, run_chatbot)

            last_msg = result["messages"][-1]
            if hasattr(last_msg, "content") and last_msg.content:
                text = last_msg.content
                if isinstance(text, list):
                    text = "".join(
                        block.get("text", "") if isinstance(block, dict) else str(block)
                        for block in text
                    )
                yield f"data: {json.dumps({'token': text})}\n\n"

            yield f"data: {json.dumps({'done': True, 'thread_id': thread_id})}\n\n"

        except Exception as e:
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'error': str(e) or repr(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
 
# ----------------------------------------------------------------------------------
# PDF upload — ingest into FAISS for a given thread (creates thread_id if missing)
# ----------------------------------------------------------------------------------


@app.post("/upload-pdf")
async def upload_pdf(
    file: UploadFile = File(...),
    thread_id: Optional[str] = Form(default=None),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    thread_id = thread_id or str(uuid.uuid4())
    file_bytes = await file.read()

    try:
        loop = asyncio.get_running_loop()
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            summary = await asyncio.wait_for(
                loop.run_in_executor(
                    pool,
                    partial(ingest_pdf, file_bytes, thread_id=thread_id, filename=file.filename)
                ),
                timeout=120.0
            )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="PDF processing timed out. Please try again.")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {e}")

    return {
        "thread_id": thread_id,
        "message": f"'{summary['filename']}' indexed successfully.",
        **summary,
    }
# ----------------------------------------------------------------------------------
# Thread management
# ----------------------------------------------------------------------------------
@app.get("/threads")
async def list_threads():
    try:
        threads = await retriev_thread()
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to fetch threads: {e}")
    return {"threads": threads, "count": len(threads)}

@app.get("/threads/{thread_id}/messages")
async def get_thread_messages(thread_id: str):
    try:
        config = {"configurable": {"thread_id": thread_id}}
        
        loop = asyncio.get_running_loop()
        checkpoint = await loop.run_in_executor(
            None,
            lambda: run_async_with_reconnect(
                lambda: checkpointer.aget(config)
            )
        )
        
        if not checkpoint or not checkpoint.get("channel_values"):
            return {"messages": []}
        messages = checkpoint["channel_values"].get("messages", [])
        result = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                result.append({"role": "user", "text": msg.content})
            elif isinstance(msg, AIMessage) and msg.content:
                result.append({"role": "assistant", "text": msg.content})
        return {"messages": result}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    


@app.get("/threads/{thread_id}/has-pdf")
async def thread_has_pdf(thread_id: str):
    from langgraph_bakend import _THREAD_RETRIEVERS, _THREAD_METADATA
    has_pdf = str(thread_id) in _THREAD_RETRIEVERS and len(_THREAD_RETRIEVERS[str(thread_id)]) > 0
    metas = _THREAD_METADATA.get(str(thread_id), [])
    filename = metas[0].get("filename", None) if metas else None
    return {"has_pdf": has_pdf, "filename": filename}

@app.delete("/threads/{thread_id}", response_model=ThreadDeleteResponse)
async def remove_thread(thread_id: str):
    try:
        await delete_thread(thread_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete thread: {e}")
    return ThreadDeleteResponse(thread_id=thread_id, deleted=True)