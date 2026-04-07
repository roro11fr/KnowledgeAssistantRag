"""
test_connection.py — Optional utility to verify API keys are configured and reachable.

Uses OpenAI only (models.list — no tokens billed).

Usage:
    python tools/test_connection.py
"""

import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def check_openai():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return False, "OPENAI_API_KEY not set"
    try:
        client = OpenAI(api_key=api_key)
        client.models.list()
        return True, "OK"
    except Exception as e:
        return False, str(e)


def main():
    results = {"OpenAI (embeddings + LLM)": check_openai()}

    all_ok = True
    for name, (ok, msg) in results.items():
        status = "OK" if ok else "FAIL"
        print(f"  [{status}] {name}: {msg}")
        if not ok:
            all_ok = False

    if not all_ok:
        print("\nCheck failed. Copy .env.example to .env and fill in your OPENAI_API_KEY.")
        sys.exit(1)
    else:
        print("\nAll connections OK. Ready to ingest and query.")


if __name__ == "__main__":
    main()
