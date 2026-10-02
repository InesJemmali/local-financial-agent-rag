"""
api.py  (step 1: the smallest useful API)

Run it:
  uvicorn api:app --reload --port 8000
Then open http://localhost:8000/docs in a browser.

Nothing here touches the agent yet. This step only proves that:
  1. a web server can run your Python code when a request arrives
  2. that code can reach Ollama
"""

import requests
from fastapi import FastAPI

from step_chat import MODEL, OLLAMA_HOST

# The application object. uvicorn looks for this variable ("api:app" = file api.py, variable app).
app = FastAPI(title="Local Financial Analyst Agent")


@app.get("/")
def root():
    # FastAPI turns the returned dict into JSON for you.
    return {"message": "The agent API is running. Open /docs to try it."}


@app.get("/health")
def health():
    """Is Ollama reachable, and which models has it pulled? Does not call the LLM."""
    try:
        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        r.raise_for_status()
        pulled = [m["name"] for m in r.json().get("models", [])]
        return {"ollama": "reachable", "ollama_host": OLLAMA_HOST,
                "chat_model": MODEL, "models_pulled": pulled}
    except requests.RequestException as e:
        return {"ollama": f"unreachable ({type(e).__name__})", "ollama_host": OLLAMA_HOST,
                "chat_model": MODEL}
