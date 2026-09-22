"""
doc_tools.py
Document search (RAG) for the financial agent.
Step C1: read every PDF in documents/ and extract its text, page by page.
Step C2: split each page into chunks of a few sentences, keeping doc and page.
Step C3: embed every chunk with an Ollama embedding model and save the index.
Step C4: search_documents(query) returns the most relevant chunks with citations.
Step C5: schema, registry and prompt text so the agent can call search_documents.

Needs: pip install pypdf numpy requests
       and the embedding model on the Ollama server (ollama pull nomic-embed-text)

  python doc_tools.py chunks   # show the chunks (C2)
  python doc_tools.py build    # embed the chunks and save the index (C3)
  python doc_tools.py search "your question"   # search the index (C4)
"""

import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import requests
from pypdf import PdfReader

from step_chat import OLLAMA_HOST        # same Ollama server as the chat model

DOCS_DIR = "documents"
MAX_CHARS = 500          # target maximum chunk size
OVERLAP_SENTENCES = 1    # sentences repeated at the start of the next chunk
MIN_SENTENCE_CHARS = 25  # shorter fragments ("1.", headings) get merged into the next sentence

EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")
INDEX_DIR = Path("index")
BATCH_SIZE = 32          # chunks sent per request to the embedding endpoint

# nomic-embed-text was trained with task prefixes; documents and queries get different ones
PREFIX = {"document": "search_document: ", "query": "search_query: "}

TOP_K = 3                # chunks returned per search
MIN_SCORE = 0.50         # best score below this = "the documents probably don't cover it"
                         # a starting guess: calibrate it on your own queries

_cache = {"mtime": None, "vectors": None, "chunks": None}   # index loaded once, reused


def load_pdfs(folder: str = DOCS_DIR) -> list[dict]:
    """Return one dict per page: {"doc": file name, "page": page number, "text": page text}."""
    folder_path = Path(folder)
    if not folder_path.is_dir():
        raise FileNotFoundError(f"Folder '{folder}' not found. Run python make_docs.py first.")

    pages = []
    for pdf_path in sorted(folder_path.glob("*.pdf")):
        reader = PdfReader(pdf_path)
        for number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if not text:
                # Scanned PDFs contain images, not text. They would need OCR.
                print(f"WARNING: no text on {pdf_path.name} page {number} (scanned page?)")
                continue
            pages.append({"doc": pdf_path.name, "page": number, "text": text})
    return pages


def _clean(text: str) -> str:
    """Join layout line breaks into spaces: PDF lines are not sentences."""
    return re.sub(r"\s+", " ", text).strip()


def _split_sentences(text: str) -> list[str]:
    """Split after . ! ? when the next word starts with a capital letter or a section number."""
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z]|\d+\.\s)", text)
    sentences, carry = [], ""
    for part in parts:
        part = (carry + " " + part).strip() if carry else part
        if len(part) < MIN_SENTENCE_CHARS:   # "1." or a short heading: glue it to the next one
            carry = part
            continue
        sentences.append(part)
        carry = ""
    if carry:                                 # leftover fragment at the end of the page
        if sentences:
            sentences[-1] += " " + carry
        else:
            sentences.append(carry)
    return sentences


def chunk_pages(pages: list[dict]) -> list[dict]:
    """Split every page into chunks of whole sentences. Chunks never cross pages."""
    chunks = []
    for p in pages:
        sentences = _split_sentences(_clean(p["text"]))
        current, n = [], 0
        for sentence in sentences:
            if current and len(" ".join(current + [sentence])) > MAX_CHARS:
                n += 1
                chunks.append(_make_chunk(p, n, current))
                current = current[-OVERLAP_SENTENCES:] if OVERLAP_SENTENCES else []
            current.append(sentence)
        if current:
            n += 1
            chunks.append(_make_chunk(p, n, current))
    return chunks


def _make_chunk(page: dict, n: int, sentences: list[str]) -> dict:
    return {"id": f"{page['doc']}#p{page['page']}-c{n}",
            "doc": page["doc"], "page": page["page"],
            "text": " ".join(sentences)}


def embed(texts: list[str], kind: str) -> np.ndarray:
    """Embed texts with Ollama. kind is "document" or "query". Returns unit-length rows."""
    prefix = PREFIX[kind]
    vectors = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = [prefix + t for t in texts[start:start + BATCH_SIZE]]
        resp = requests.post(f"{OLLAMA_HOST}/api/embed",
                             json={"model": EMBED_MODEL, "input": batch}, timeout=120)
        resp.raise_for_status()
        vectors.extend(resp.json()["embeddings"])
    matrix = np.array(vectors, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-12, None)   # unit length: cosine similarity = dot product


def build_index(folder: str = DOCS_DIR) -> None:
    """Extract, chunk, embed and save. Run again whenever the documents change."""
    t0 = time.time()
    chunks = chunk_pages(load_pdfs(folder))
    vectors = embed([c["text"] for c in chunks], kind="document")

    INDEX_DIR.mkdir(exist_ok=True)
    np.save(INDEX_DIR / "vectors.npy", vectors)
    meta = {"embed_model": EMBED_MODEL, "dim": int(vectors.shape[1]),
            "max_chars": MAX_CHARS, "overlap_sentences": OVERLAP_SENTENCES,
            "built_at": time.strftime("%Y-%m-%d %H:%M:%S"), "chunks": chunks}
    (INDEX_DIR / "chunks.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"embedded {len(chunks)} chunks with {EMBED_MODEL} in {time.time() - t0:.1f}s")
    print(f"vectors: {vectors.shape}, row norms {np.linalg.norm(vectors, axis=1).min():.3f}"
          f" to {np.linalg.norm(vectors, axis=1).max():.3f}")
    print(f"saved {INDEX_DIR / 'vectors.npy'} and {INDEX_DIR / 'chunks.json'}")


def _load_index():
    """Load the index from disk, reusing it until the files change."""
    vec_path, meta_path = INDEX_DIR / "vectors.npy", INDEX_DIR / "chunks.json"
    if not vec_path.exists() or not meta_path.exists():
        raise FileNotFoundError("No document index found. Run: python doc_tools.py build")

    mtime = vec_path.stat().st_mtime
    if _cache["mtime"] != mtime:                      # first call, or the index was rebuilt
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        vectors = np.load(vec_path)
        if meta["embed_model"] != EMBED_MODEL:
            raise ValueError(f"Index was built with '{meta['embed_model']}' but queries use "
                             f"'{EMBED_MODEL}'. Rebuild the index or set EMBED_MODEL.")
        if vectors.shape != (len(meta["chunks"]), meta["dim"]):
            raise ValueError("vectors.npy and chunks.json do not match. Rebuild the index.")
        _cache.update(mtime=mtime, vectors=vectors, chunks=meta["chunks"])
    return _cache["vectors"], _cache["chunks"]


def search_documents(query: str, top_k: int = TOP_K) -> str:
    """Return the chunks most similar to the query, with doc, page and score, as JSON."""
    query = query.strip()
    if not query:
        raise ValueError("Empty query.")
    top_k = max(1, min(int(top_k), 5))                # the model may send "3" or 50

    vectors, chunks = _load_index()
    scores = vectors @ embed([query], kind="query")[0]   # cosine similarity with every chunk
    best = np.argsort(scores)[::-1][:top_k]              # indices of the highest scores

    results = [{"id": chunks[i]["id"], "doc": chunks[i]["doc"], "page": chunks[i]["page"],
                "score": round(float(scores[i]), 3), "text": chunks[i]["text"]}
               for i in best]

    out = {"source": "documents", "query": query, "result_count": len(results),
           "results": results}
    if results[0]["score"] < MIN_SCORE:
        out["note"] = ("No strong match: the documents may not cover this question. "
                       "Say so instead of answering from these passages.")
    return json.dumps(out)


# ---------------------------------------------------------------- what the agent sees

# Offered to the agent only when an index exists (python doc_tools.py build)
DOCS_ENABLED = (INDEX_DIR / "vectors.npy").exists() and (INDEX_DIR / "chunks.json").exists()


def _doc_names() -> str:
    """Names of the indexed documents, so the schema tells the model what it can search."""
    if not DOCS_ENABLED:
        return "none"
    meta = json.loads((INDEX_DIR / "chunks.json").read_text(encoding="utf-8"))
    return ", ".join(sorted({c["doc"] for c in meta["chunks"]}))


DOC_REGISTRY = {"search_documents": search_documents}

DOC_SCHEMAS = [{
    "type": "function",
    "function": {
        "name": "search_documents",
        "description": (
            "Search the bank's internal documents for rules, thresholds, targets and "
            f"procedures. Indexed documents: {_doc_names()}. Use it whenever a question "
            "mentions a policy, rule, limit, threshold, target, procedure, or asks whether "
            "the data complies with one. Never use it for numbers about the transactions; "
            "those come from the dataset tools. Returns passages with document name, page "
            "and a similarity score."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to look for, as a short question or key phrase, "
                                   "e.g. 'review threshold for Travel transactions'.",
                }
            },
            "required": ["query"],
        },
    },
}]

DOC_SYSTEM = """INTERNAL DOCUMENTS

You also have search_documents, which searches the bank's internal policies, procedures and reports. Each source has one role: numbers about the transactions come only from the dataset tools; rules, thresholds, targets and procedures come only from search_documents. Never take a threshold or target from your own knowledge.

A question about a rule, policy, target or procedure is not a missing-column case: search the documents for it.

When a question compares the data with a rule or target, first call search_documents to get the rule, then call the dataset tools to compute the figure, then compare the two.

Cite every fact from the documents as (document name, page N). If a search result has a "note" saying there is no strong match, say the documents do not cover the question and do not answer from those passages.

When your answer uses documents, write what the data shows first, then "From our documents:" with the citations, then "External context:" if you also used the web."""


def _show_chunks() -> None:
    chunks = chunk_pages(load_pdfs())
    sizes = [len(c["text"]) for c in chunks]
    print(f"{len(chunks)} chunks (min {min(sizes)}, max {max(sizes)}, "
          f"mean {sum(sizes) // len(sizes)} chars)\n")
    for c in chunks:
        print(f"[{c['id']}]  {len(c['text'])} chars")
        print(f"    {c['text']}\n")


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "chunks"
    if command == "build":
        build_index()
    elif command == "chunks":
        _show_chunks()
    elif command == "search":
        out = json.loads(search_documents(" ".join(sys.argv[2:])))
        print(f"query: {out['query']}\n")
        for r in out["results"]:
            print(f"{r['score']:.3f}  {r['id']}\n       {r['text'][:150]}...\n")
        if "note" in out:
            print(f"NOTE: {out['note']}")
    else:
        print('usage: python doc_tools.py [chunks | build | search "question"]')