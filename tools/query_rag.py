"""
query_rag.py — Answer a question using documents indexed in ChromaDB.

Pipeline (2 API calls, both OpenAI):
  1. Embed question via OpenAI text-embedding-3-small
  2. Retrieve top-k chunks from ChromaDB (local)
  3. Generate grounded answer via OpenAI gpt-4o-mini

Usage:
    python tools/query_rag.py --question "What is the refund policy?"
    python tools/query_rag.py --question "..." --top-k 5
"""

import argparse
import os
import re
import sys

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "documents"
EMBED_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
DEFAULT_TOP_K = 5

# ---------------------------------------------------------------------------
# Intent detection
# ---------------------------------------------------------------------------

_TROUBLESHOOT_RE = re.compile(
    r"\b(why|not working|problem|issue|broken|fail|error|weak|won'?t|doesn'?t|"
    r"can'?t|cannot|no suction|won'?t start|does not|doesn'?t work)\b",
    re.IGNORECASE,
)
_HOWTO_RE = re.compile(
    r"\b(how|steps?|guide|instruction|procedure|do i|can i|show me|walk me)\b",
    re.IGNORECASE,
)


def detect_intent(question: str) -> str:
    if _TROUBLESHOOT_RE.search(question):
        return "troubleshooting"
    if _HOWTO_RE.search(question):
        return "howto"
    return "general"


# ---------------------------------------------------------------------------
# System prompts (one per intent)
# ---------------------------------------------------------------------------

_PROMPT_TROUBLESHOOTING = """You are a precise technical support assistant. Answer strictly from the provided documentation context.

Format your answer EXACTLY using these section headers (no markdown bold, plain text):

Problem:
<one concise sentence describing the problem>

Possible causes:
- <cause> (page N)
- <cause> (page N)

Recommended actions:
- <action> (page N)
- <action> (page N)

Rules:
- Use ONLY information from the provided context.
- After each cause or action, cite the page number in parentheses: (page N).
- Use the page numbers shown in the context headers — e.g. "Source: file.pdf, page 14" means cite (page 14).
- If the context does not contain enough information, write under Problem: "The documentation does not cover this specific issue." and omit the other sections.
- Be direct and specific. Do not use vague words like "possibly" or "maybe".
"""

_PROMPT_HOWTO = """You are a clear technical guide assistant. Answer strictly from the provided documentation context.

Format your answer as numbered steps:

1. <step> (page N)
2. <step> (page N)
...

Rules:
- Use ONLY information from the provided context.
- After each step, cite the page number in parentheses: (page N).
- Use the page numbers shown in the context headers.
- If the context does not contain enough information, say: "The documentation does not include steps for this task."
- Be concise and action-oriented.
"""

_PROMPT_GENERAL = """You are a precise documentation assistant. Answer strictly from the provided context.

Rules:
- Use ONLY information from the provided context.
- After each key claim, cite the page number: (page N).
- Use page numbers shown in the context headers.
- If the context does not contain enough information, say so explicitly.
- Be direct and confident. Avoid hedging language like "possibly" or "it might be".
- Keep the answer concise.
"""

_PROMPTS = {
    "troubleshooting": _PROMPT_TROUBLESHOOTING,
    "howto": _PROMPT_HOWTO,
    "general": _PROMPT_GENERAL,
}


# ---------------------------------------------------------------------------
# Pipeline functions
# ---------------------------------------------------------------------------

def embed_question(question: str, client: OpenAI) -> list[float]:
    response = client.embeddings.create(model=EMBED_MODEL, input=question)
    return response.data[0].embedding


def retrieve_chunks(collection, embedding: list[float], top_k: int) -> list[dict]:
    results = collection.query(
        query_embeddings=[embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text": doc,
            "source": meta.get("source", "unknown"),
            "distance": dist,
            "page": meta.get("page", 0),
        })
    return chunks


def build_context(chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        page = chunk.get("page")
        page_str = f", page {page}" if page else ""
        parts.append(f"[{i}] Source: {chunk['source']}{page_str}\n{chunk['text']}")
    return "\n\n---\n\n".join(parts)


def generate_answer(client: OpenAI, question: str, context: str, intent: str = "general") -> str:
    system_prompt = _PROMPTS.get(intent, _PROMPT_GENERAL)
    user_message = f"Context:\n{context}\n\nQuestion: {question}"
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=0.0,
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Query the RAG index.")
    parser.add_argument("--question", required=True, help="Question to answer")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Number of sections to retrieve")
    args = parser.parse_args()

    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        print("Error: OPENAI_API_KEY not set. Copy .env.example to .env and fill in your key.")
        sys.exit(1)

    openai_client = OpenAI(api_key=openai_key)
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

    try:
        collection = chroma_client.get_collection(name=COLLECTION_NAME)
    except Exception:
        print(f"Error: No indexed documents found in '{CHROMA_PATH}'. Run ingest_docs.py first.")
        sys.exit(1)

    intent = detect_intent(args.question)
    print(f"Intent: {intent}")
    print("Embedding question...")
    embedding = embed_question(args.question, openai_client)

    print(f"Retrieving top-{args.top_k} sections...")
    chunks = retrieve_chunks(collection, embedding, args.top_k)

    if not chunks:
        print("\nNo relevant content found in the indexed documents for this question.")
        sys.exit(0)

    print(f"Found {len(chunks)} section(s). Generating answer...\n")
    context = build_context(chunks)
    answer = generate_answer(openai_client, args.question, context, intent)

    print("=" * 60)
    print(answer)
    print("=" * 60)
    print(f"\nSources: {', '.join(sorted({c['source'] for c in chunks}))}")


if __name__ == "__main__":
    main()
