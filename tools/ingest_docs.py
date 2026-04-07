"""
ingest_docs.py — Parse, chunk, embed, and store documents into ChromaDB.

Supports: PDF (.pdf), Word (.docx)
Embeddings: OpenAI text-embedding-3-small
Vector store: ChromaDB (local, persisted in chroma_db/)

Usage:
    python tools/ingest_docs.py --source <path/to/docs>
    python tools/ingest_docs.py --source <path/to/docs> --reset   # clear index first

Skips documents already indexed (SHA-256 hash check).
Page numbers are stored per chunk for PDF files.
"""

import argparse
import hashlib
import os
import sys
from pathlib import Path

import chromadb
import fitz  # pymupdf
from docx import Document
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "documents"
CHUNK_SIZE = 500       # characters
CHUNK_OVERLAP = 50     # characters
EMBED_MODEL = "text-embedding-3-small"
BATCH_SIZE = 100       # OpenAI supports up to 2048 inputs per call


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def fix_pdf_text(text: str) -> str:
    """Fix common PDF text extraction artifacts.

    PyMuPDF sometimes extracts ligature characters (fi, fl, ff…) as raw UTF-8
    bytes decoded as Latin-1, producing garbled surrogates. Re-encoding as
    Latin-1 and decoding as UTF-8 restores the original characters.
    Standard Unicode ligature codepoints are also normalised to ASCII.
    """
    # Standard Unicode ligatures → plain ASCII
    for lig, rep in [('\ufb00','ff'),('\ufb01','fi'),('\ufb02','fl'),
                     ('\ufb03','ffi'),('\ufb04','ffl'),('\ufb05','st'),('\ufb06','st')]:
        text = text.replace(lig, rep)
    # Re-encode bytes that were misread as Latin-1 instead of UTF-8
    try:
        text = text.encode('latin-1', errors='surrogatepass').decode('utf-8', errors='replace')
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    return text


def parse_pdf(path: Path) -> list[tuple[str, int]]:
    """Returns list of (page_text, 1-indexed page number) for non-empty pages."""
    doc = fitz.open(str(path))
    pages = []
    for page in doc:
        text = fix_pdf_text(page.get_text().strip())
        if text:
            pages.append((text, page.number + 1))
    return pages


def parse_docx(path: Path) -> str:
    doc = Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start += chunk_size - overlap
    return [c for c in chunks if c]


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def embed_texts(texts: list[str], client: OpenAI) -> list[list[float]]:
    """Embed texts in batches — OpenAI supports up to 2048 per call."""
    all_embeddings = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        response = client.embeddings.create(model=EMBED_MODEL, input=batch)
        all_embeddings.extend([e.embedding for e in response.data])
    return all_embeddings


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Ingest documents into ChromaDB.")
    parser.add_argument("--source", required=True, help="Directory containing documents to ingest")
    parser.add_argument("--reset", action="store_true", help="Clear the index before ingesting")
    args = parser.parse_args()

    source_dir = Path(args.source)
    if not source_dir.is_dir():
        print(f"Error: '{source_dir}' is not a valid directory.")
        sys.exit(1)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not set. Copy .env.example to .env and fill in your key.")
        sys.exit(1)

    openai_client = OpenAI(api_key=api_key)
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

    if args.reset:
        try:
            chroma_client.delete_collection(name=COLLECTION_NAME)
            print("Cleared existing index.")
        except Exception:
            pass

    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

    existing_meta = collection.get(include=["metadatas"])["metadatas"]
    indexed_hashes = {m["doc_hash"] for m in existing_meta if m and "doc_hash" in m}

    files = [f for f in source_dir.iterdir() if f.suffix.lower() in {".pdf", ".docx"}]
    if not files:
        print(f"No PDF or DOCX files found in '{source_dir}'.")
        sys.exit(0)

    total_new_chunks = 0

    for file_path in files:
        doc_hash = file_hash(file_path)
        if doc_hash in indexed_hashes:
            print(f"  [skip] already indexed: {file_path.name}")
            continue

        print(f"  [ingest] {file_path.name}")
        suffix = file_path.suffix.lower()

        if suffix == ".pdf":
            pages = parse_pdf(file_path)
            chunks, page_nums = [], []
            for page_text, page_num in pages:
                pc = chunk_text(page_text)
                chunks.extend(pc)
                page_nums.extend([page_num] * len(pc))
        elif suffix == ".docx":
            text = parse_docx(file_path)
            chunks = chunk_text(text)
            page_nums = [0] * len(chunks)  # 0 = page unknown for docx
        else:
            print(f"  [skip] unsupported file type: {file_path.name}")
            continue

        if not chunks:
            continue

        print(f"    {len(chunks)} sections — embedding...")
        embeddings = embed_texts(chunks, openai_client)

        ids = [f"{doc_hash}_{i}" for i in range(len(chunks))]
        metadatas = [
            {"source": file_path.name, "chunk_index": i, "doc_hash": doc_hash, "page": page_nums[i]}
            for i in range(len(chunks))
        ]

        collection.add(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)
        total_new_chunks += len(chunks)
        print(f"    stored {len(chunks)} sections.")

    print(f"\nDone. {total_new_chunks} new sections added.")


if __name__ == "__main__":
    main()
