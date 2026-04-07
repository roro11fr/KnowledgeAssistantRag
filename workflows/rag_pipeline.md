# Workflow: RAG Pipeline

## Objective
Answer user questions accurately using documentation as the source of truth. Retrieve relevant content from the vector store and generate a grounded response — never answer from memory alone.

## Required Inputs
- A directory of source documents (PDF and/or .docx files)
- A user question (natural language)

## Tools
- `tools/ingest_docs.py` — ingests and indexes documents
- `tools/query_rag.py` — answers a question using the index

## Steps

### Phase 1: Ingest (run once per document set, or when docs change)

1. Confirm the source directory exists and contains supported files (`.pdf`, `.docx`)
2. Run ingestion:
   ```bash
   python tools/ingest_docs.py --source <path/to/docs>
   ```
3. The tool will:
   - Parse each file (PDF via `pymupdf`, Word via `python-docx`)
   - Split text into overlapping chunks (~500 tokens, 50-token overlap)
   - Compute a SHA-256 hash per document — skip if already indexed in ChromaDB
   - Embed each chunk via Gemini `text-embedding-004` (free tier)
   - Store chunks + embeddings in the local `chroma_db/` collection
4. Confirm output: tool prints count of new chunks added and skipped docs

### Phase 2: Query

1. Run a query:
   ```bash
   python tools/query_rag.py --question "Your question here"
   ```
2. The tool will:
   - Embed the question via Gemini (1 API call)
   - Retrieve top-5 most similar chunks from ChromaDB (local, no API call)
   - Send chunks + question to OpenAI `gpt-4o-mini` with a strict grounding prompt (1 API call)
   - Print the answer with source references (filename + chunk index)

## Expected Outputs
- A factual answer grounded in the retrieved documentation
- Source references so the answer can be verified

## Edge Cases

| Situation | Action |
|-----------|--------|
| No relevant chunks found (low similarity score) | Tool warns "no relevant content found" and declines to answer |
| Document already indexed | Skipped silently (hash match) — no duplicate embeddings |
| Unsupported file type | Tool prints a warning and skips the file |
| API key missing | Tool exits with a clear error message pointing to `.env.example` |
| Rate limit hit on Gemini free tier | Add `time.sleep(1)` between embed calls; document here if adjusted |

## Notes
- ChromaDB data is stored in `chroma_db/` at the project root (gitignored)
- Gemini `text-embedding-004` produces 768-dimensional embeddings
- Model can be swapped from `gpt-4o-mini` to `gpt-4o` in `tools/query_rag.py` for higher quality
- Re-run ingest whenever new documents are added to the source directory
