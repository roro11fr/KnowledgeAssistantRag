# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

This is a **RAG Agent Based on Documentation** project built on the **WAT framework** (Workflows, Agents, Tools). The goal is to build a retrieval-augmented generation agent that answers questions based on ingested documentation. The architecture separates probabilistic AI reasoning from deterministic code execution.

## WAT Architecture

**Layer 1: Workflows (`workflows/`)** — Markdown SOPs defining objectives, required inputs, which tools to use, expected outputs, and edge case handling. Do not create or overwrite workflow files without explicit user approval.

**Layer 2: Agents** — Your role. Read the relevant workflow, execute tools in sequence, handle failures gracefully, and ask clarifying questions when needed. Never try to do everything directly — delegate execution to tools.

**Layer 3: Tools (`tools/`)** — Python scripts for deterministic execution: API calls, data transformations, file ops, database queries. Always check `tools/` before writing anything new.

## Operating Rules

- **Always check `tools/` first.** Only create new scripts when nothing exists for the task.
- **Before re-running any tool that makes paid API calls or uses credits, confirm with the user.**
- **On error:** read the full trace, fix the script, retest, then update the workflow with what you learned (rate limits, timing quirks, unexpected behavior).
- **Workflows must stay current.** When you discover better methods or constraints, update the workflow file.

## File Structure

```
.tmp/              # Temporary/intermediate files — disposable, regenerated as needed
tools/             # Python scripts (deterministic execution)
workflows/         # Markdown SOPs
.env               # API keys and credentials (only place for secrets)
credentials.json   # Google OAuth (gitignored)
token.json         # Google OAuth token (gitignored)
```

Deliverables go to cloud services (Google Sheets, Slides, etc.) — not local files. Local files are for processing only.

## Development Commands

As tools are added, run them directly:

```bash
python tools/<script_name>.py
```

Load environment variables before running tools that need API keys:

```bash
# On Windows (bash)
export $(cat .env | xargs) && python tools/<script_name>.py
```

## Self-Improvement Loop

When something breaks:
1. Identify what broke
2. Fix the tool
3. Verify the fix
4. Update the workflow with the new approach
