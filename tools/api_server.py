"""
api_server.py — FastAPI server exposing the RAG pipeline as an HTTP API.

Endpoints:
  POST /query   — answer a question using indexed documents
  GET  /status  — health check + doc count in the index
  GET  /         — serve the frontend

Usage:
    python tools/api_server.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import chromadb
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from openai import OpenAI
from pydantic import BaseModel

from query_rag import (
    CHROMA_PATH,
    COLLECTION_NAME,
    embed_question,
    retrieve_chunks,
    build_context,
    generate_answer,
    detect_intent,
    DEFAULT_TOP_K,
)

load_dotenv()

app = FastAPI(title="Knowledge Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
DOCS_DIR = Path(__file__).parent.parent / "docs"

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
if DOCS_DIR.exists():
    app.mount("/docs", StaticFiles(directory=str(DOCS_DIR)), name="docs")

# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

_openai_key = os.getenv("OPENAI_API_KEY")
if not _openai_key:
    raise RuntimeError("OPENAI_API_KEY not set. Copy .env.example to .env.")

_openai_client = OpenAI(api_key=_openai_key)
_chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)


def get_collection():
    try:
        return _chroma_client.get_collection(name=COLLECTION_NAME)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str
    top_k: int = DEFAULT_TOP_K


class SourceRef(BaseModel):
    source: str
    excerpt: str
    distance: float
    page: int = 0
    doc_url: str = ""


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceRef]
    confidence: str = "Low"   # "High" | "Medium" | "Low"
    intent: str = "general"   # "troubleshooting" | "howto" | "general"


class StatusResponse(BaseModel):
    status: str
    indexed_chunks: int


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def serve_frontend():
    index = FRONTEND_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"message": "API running. Open frontend/index.html in your browser."}


@app.get("/status", response_model=StatusResponse)
def status():
    collection = get_collection()
    count = collection.count() if collection else 0
    return StatusResponse(
        status="ready" if (collection and count > 0) else "no_index",
        indexed_chunks=count,
    )


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Please enter a question.")

    collection = get_collection()
    if not collection or collection.count() == 0:
        raise HTTPException(
            status_code=503,
            detail="No documents have been loaded yet. Please contact your administrator.",
        )

    intent = detect_intent(req.question)

    embedding = embed_question(req.question, _openai_client)
    chunks = retrieve_chunks(collection, embedding, req.top_k)

    if not chunks:
        raise HTTPException(
            status_code=503,
            detail="No documents have been loaded yet. Please contact your administrator.",
        )

    context = build_context(chunks)
    answer = generate_answer(_openai_client, req.question, context, intent)

    # Confidence: based on the best (lowest) distance in top results
    best_dist = min(c["distance"] for c in chunks)
    if best_dist < 0.90:
        confidence = "High"
    elif best_dist < 1.10:
        confidence = "Medium"
    else:
        confidence = "Low"

    from urllib.parse import quote

    def _doc_url(source: str, page: int) -> str:
        base = f"/docs/{quote(source)}"
        return f"{base}#page={page}" if page else base

    sources = [
        SourceRef(
            source=c["source"],
            excerpt=c["text"][:600].strip(),
            distance=round(c["distance"], 4),
            page=c.get("page", 0),
            doc_url=_doc_url(c["source"], c.get("page", 0)),
        )
        for c in chunks
    ]
    return QueryResponse(answer=answer, sources=sources, confidence=confidence, intent=intent)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)
