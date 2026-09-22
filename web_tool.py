import json
import sys
from ddgs import DDGS
from ddgs.exceptions import DDGSException, RatelimitException, TimeoutException
import os
import time

MAX_RESULTS = 5
SNIPPET_CHARS = 300   # cap each snippet so results stay small in the context
COOLDOWN_S = 120      # after a network failure, skip the web for this long

_web_down_until = 0.0  # time until which web_search refuses without touching the network

WEB_MODE = os.environ.get("WEB_MODE", "off").strip().lower()
if WEB_MODE not in {"open", "off"}:
    raise ValueError(f"Invalid WEB_MODE '{WEB_MODE}'. Use 'open' or 'off'.")


def web_search(query: str) -> str:
    """Search the web. Returns titles, URLs and short snippets as JSON."""
   

    global _web_down_until

    query = query.strip()
    if not query:
        raise ValueError("Empty query.")

    # Circuit breaker: the web failed recently, so fail fast without a network call
    if time.monotonic() < _web_down_until:
        raise RuntimeError("Web search is unavailable (it failed moments ago). Do not call "
                           "web_search again. Answer without web context and tell the user.")

    try:
        raw = DDGS(timeout=10).text(query, max_results=MAX_RESULTS) or []
    except Exception as e:
        if isinstance(e, DDGSException) and "no results" in str(e).lower():
            raw = []           # ddgs raises when nothing is found; that is not a failure
        else:
            _web_down_until = time.monotonic() + COOLDOWN_S      # open the breaker
            raise RuntimeError(f"Web search is unreachable ({type(e).__name__}). This machine "
                               "may have no internet access. Do not retry. Tell the user "
                               "the web could not be reached.") from e

    results = [
        {"id": i,
         "title": r.get("title", ""),
         "url": r.get("href", ""),
         "snippet": r.get("body", "")[:SNIPPET_CHARS]}
        for i, r in enumerate(raw, 1)
    ]

    return json.dumps({"source": "web", "query": query,"result_count": len(results), "results": results})



WEB_REGISTRY = {"web_search": web_search}

WEB_SCHEMAS = [{
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the public internet for context the dataset cannot contain: news, "
            "economic conditions, interest rates, regulation, seasonal patterns, or the "
            "meaning of a financial term. Never use it for numbers about the user's own "
            "data; those come from the dataset tools. Never put customer IDs, amounts or "
            "other values from the dataset into the query. Returns titles, URLs, snippets."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Short search-engine query, 3 to 8 words, with names "
                                   "and dates, e.g. 'Bank of Canada rate decision 2025'.",
                }
            },
            "required": ["query"],
        },
    },
}]

WEB_SYSTEM = """WEB CONTEXT

You also have web_search. The dataset is the source of truth for every number about the user's data. The web only supplies outside context.

A question about causes, events or conditions outside the data (why something happened, what the economy was doing, what a term means) is not a missing-column case.Answer the data part with dataset tools, then call web_search for the context part. Saying you will search is not a search.

When your answer uses web results, write it in two parts: first what the data shows, then "External context:" with each web fact followed by its URL. Never present a web fact as something the data shows. Use at most 2 web searches per question. If nothing relevant turns up, say so instead of guessing."""

WEB_TURN_HINT = ("If part of the question needs outside context the data cannot contain, "
                 "call web_search for it in this same response. Writing that you will "
                 "search does not perform a search; only a tool call does.")


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) or "latest news on AI"
    print(json.dumps(json.loads(web_search(query)), indent=2))



    