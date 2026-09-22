# Local Financial Analyst Agent

A tool-calling AI agent that answers questions about bank transaction data by combining three sources: a transaction dataset (CSV or Excel), internal bank documents (policies, procedures, reports) and the web. It runs entirely on a local open-weight model through Ollama, so the data and documents never leave the machine.

Built from scratch in Python, without agent frameworks, to show exactly how the agent loop, function calling and retrieval (RAG) work underneath.

## What it does

An analyst asks a question in plain English. The model decides which tools it needs, the application runs them, and the model writes an answer grounded in the results, with the source cited.

```text
you> What share of total spending was Travel, and did it meet the 2025 target?

 [step 1.1] group_aggregate({'group_by': 'Merchant_Category', 'how': 'sum', 'value_column': 'Amount', ...})
       -> {"result": {"Grocery": 133573.74, "Travel": 126948.15, "Utilities": 123823.35, "Dining": 117313.42}}
 [step 1.2] search_documents({'query': "2025 target for Travel's share of total spending"})
       -> {"results": [{"doc": "spending_report_2025.pdf", "page": 1, "score": 0.772, ...}]}

bot> The share of total spending for Travel is (126,948.15 / 501,658.66) * 100 ≈ 25.3%.
     From our documents (spending_report_2025.pdf, page 1), the 2025 target for Travel's
     share is less than 25%. Travel's share did not meet the target.
```

The number comes from the dataset. The target comes from the policy document. The model's job is to connect them.

## Why this design fits a banking context

- **Data stays local.** The language model and the embedding model both run through Ollama on your own hardware. No transaction data or internal document is sent to an external AI provider.
- **Numbers come from code, not from the model.** Every figure is computed by pandas inside a tool. The model interprets results; it does not invent them.
- **Answers cite their sources.** Document-based answers name the file and page, so a reviewer can check the passage.
- **Every step is visible.** Each tool call, its arguments and its result are printed, which gives a readable audit trail of how an answer was produced.
- **Failures are reported, not hidden.** If a tool fails or the documents do not cover a question, the agent says so instead of guessing. For example, asked about a crypto policy that does not exist, it answers that the documents do not address it.
- **Careful wording on risk.** Anomaly detection returns statistical outliers with an explicit caveat that an outlier is not evidence of fraud.

## Architecture

```text
USER QUESTION
     ↓
agent_conversation.py      application: keeps history, runs the loop, executes tools
     ↓            ↑
Ollama (qwen3:14b)          LLM: decides which tool to call, writes the final answer
     ↓
TOOLS
 ├── tools.py         dataset tools: inspect, group/filter aggregates, anomaly detection (pandas)
 ├── doc_tools.py     search_documents: retrieval over the PDF index (RAG)
 └── web_tools.py     web_search: external information
     ↓
RESULTS (JSON) → back to the LLM → answer with citations
```

The split of responsibilities is deliberate:

| Component | Does | Does not |
|---|---|---|
| LLM | Chooses tools and arguments, interprets results, writes the answer | Read files, run code or touch the network |
| Application | Sends messages to the model, runs the requested tool, returns the result, limits the number of steps | Decide what to answer |
| Tools | Compute on the dataset, search the index, query the web | Reason about the question |

### Document retrieval (RAG)

```text
PDFs in documents/
   ↓  pypdf: extract text per page
   ↓  split into chunks of a few sentences, keeping doc name and page
   ↓  nomic-embed-text (via Ollama): one 768-dimension vector per chunk
index/vectors.npy + index/chunks.json

At question time:
question → embedded with the same model → dot product with all chunk vectors
         → top 3 chunks with doc, page and score → given to the LLM
```

A NumPy matrix is used instead of a vector database. For a few thousand chunks it is faster and has nothing hidden. The index records which embedding model built it, and the search refuses to run if the query model differs, so mismatched vectors cannot silently return wrong passages. A minimum similarity score lets the tool report when the documents probably do not cover a question.

## Project structure

```text
.
├── agent_conversation.py   interactive agent: conversation history and the tool loop
├── step1_chat.py           connection to the Ollama server (host, chat model)
├── tools.py                dataset tools (pandas)
├── doc_tools.py            PDF loading, chunking, embedding, index build and search
├── web_tools.py            web search tool
├── make_data.py            generates a synthetic transaction dataset in data/
├── make_docs.py            generates synthetic bank documents in documents/
├── data/                   transaction datasets (CSV or Excel)
└── documents/              PDF documents for retrieval
```

`index/` is created by the build step and is not committed.

## Installation

### 1. Install Ollama

Ollama runs open-weight models locally and exposes them through an HTTP API.

- **Linux**
  ```bash
  curl -fsSL https://ollama.com/install.sh | sh
  ```
- **macOS and Windows:** download the installer from [ollama.com/download](https://ollama.com/download).

Check the installation:

```bash
ollama --version
```

### 2. Download the models

Two models are needed: one to reason and chat, one to turn text into vectors for document search.

```bash
ollama pull qwen3:14b          # chat model with tool calling (about 9 GB)
ollama pull nomic-embed-text   # embedding model for document search (about 270 MB)
```

`qwen3:14b` runs best on a GPU with at least 16 GB of memory. On a smaller machine, `qwen3:8b` also supports tool calling; change the chat model name in `step1_chat.py` accordingly.

### 3. Start the Ollama server

```bash
ollama serve
```

On macOS and Windows the desktop app starts the server automatically. Check that it runs and sees both models:

```bash
curl http://127.0.0.1:11434/api/tags
```

Both `qwen3:14b` and `nomic-embed-text` should appear in the JSON.

### 4. Get the code and install the Python dependencies

Python 3.10 or later is required.

```bash
git clone <this-repository-url>
cd <repository-folder>

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install requests pandas numpy openpyxl pypdf reportlab ddgs
```

| Package | Used for |
|---|---|
| `requests` | talking to the Ollama API |
| `pandas`, `openpyxl` | reading and analysing CSV and Excel files |
| `numpy` | the document vector index |
| `pypdf` | extracting text from PDFs |
| `reportlab` | generating the sample PDFs |
| `ddgs` | web search |

## Running the project

### Step 1. Prepare data and documents

**Option A: use the sample data (recommended for a first run).**

```bash
python make_data.py     # writes data/tx.csv (5,000 synthetic transactions)
python make_docs.py     # writes 3 PDFs into documents/
```

The generated bank, Northbridge Bank, is fictional. The documents are written around the dataset columns (`Date`, `Customer_ID`, `Account_Type`, `Transaction_Type`, `Amount`, `Country`, `Merchant_Category`), so questions can combine both:

| Document | Contains | Example question |
|---|---|---|
| `card_spending_policy.pdf` | review thresholds per category, cross-border rules, anomaly screening method | "How many Travel transactions exceed the review threshold?" |
| `fraud_review_procedure.pdf` | steps after a transaction is flagged, escalation | "What must happen after an anomaly is flagged?" |
| `spending_report_2025.pdf` | 2025 targets and expectations | "Did Travel stay below its target share?" |

The documents hold rules and targets only, never results. The agent always has to compute the actual figures from the dataset.

**Option B: use your own files.**

- Put your CSV or Excel files in `data/` and refer to them by file name in your questions.
- Put your PDFs in `documents/`. They must contain real text: scanned PDFs are images and would need OCR first. Pages without text are skipped with a warning.

### Step 2. Build the document index

Check how the PDFs are chunked (no model needed):

```bash
python doc_tools.py chunks
```

Embed the chunks and save the index (Ollama must be running):

```bash
python doc_tools.py build
```

Expected output with the sample documents:

```text
embedded 10 chunks with nomic-embed-text in 1.8s
vectors: (10, 768), row norms 1.000 to 1.000
saved index/vectors.npy and index/chunks.json
```

Test retrieval alone, without the agent:

```bash
python doc_tools.py search "When must Travel transactions be reviewed?"
```

Rebuild the index whenever you add, remove or edit a PDF.

### Step 3. Run the agent

```bash
WEB_MODE=open python agent_conversation.py
```

`WEB_MODE=open` enables the web search tool. Leave it out to keep the agent limited to the dataset and the internal documents.

Inside the conversation:

- type a question and press Enter
- `reset` clears the conversation history
- `quit` exits

## Example questions

| Question | Sources used |
|---|---|
| What is the review threshold for dining transactions? | documents |
| Does the average transaction amount for Savings accounts meet the 2025 target? | dataset + documents |
| What share of total spending was Travel, and did it meet the 2025 target? | dataset + documents |
| Find the customer with the largest anomalous transaction. | dataset |
| What does our policy say about cryptocurrency purchases? | documents (correctly reports no coverage) |
| What is the Bank of Canada's current policy rate? | web |

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_HOST` | local Ollama server | address of the Ollama server, for example `gpu-server:11434` |
| `EMBED_MODEL` | `nomic-embed-text` | embedding model for the document index |
| `WEB_MODE` | web disabled | set to `open` to allow web search |

If you change `EMBED_MODEL`, rebuild the index. The search checks the model name stored in the index and stops with an explicit error on a mismatch.

### Running the model on a separate GPU server

The agent and the model communicate only over HTTP, so they can run on different machines. Start Ollama on the GPU machine so it accepts network connections:

```bash
OLLAMA_HOST=0.0.0.0:11434 ollama serve
```

Then point the agent at it:

```bash
OLLAMA_HOST=<gpu-machine-name>:11434 python doc_tools.py build
OLLAMA_HOST=<gpu-machine-name>:11434 WEB_MODE=open python agent_conversation.py
```

Only expose the server on a trusted network.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `Connection refused` or timeout when starting | Ollama is not running, or wrong `OLLAMA_HOST` | start `ollama serve`, check with `curl <host>/api/tags` |
| `model "nomic-embed-text" not found` | embedding model not pulled on the server in use | `ollama pull nomic-embed-text`, then restart the server |
| `No document index found` | index not built yet | `python doc_tools.py build` |
| `Index was built with ... but queries use ...` | embedding model changed after the build | rebuild the index |
| `Folder 'documents' not found` | no documents yet | `python make_docs.py` or add your own PDFs |
| `WARNING: no text on ... page ...` | scanned PDF page | run OCR on the file first |

## Limitations

- The sample data and documents are synthetic and meant for demonstration.
- A local 14B model occasionally stops one step short on multi-step questions (for example, finding a threshold but not running the final filter). Asking a follow-up question usually completes the analysis.
- Retrieval uses a flat NumPy index, which suits thousands of chunks. A larger document base would call for a vector database such as FAISS or pgvector.
- There is no user authentication or role-based access. A production deployment would need both, together with persistent audit logging and human approval for any action beyond read-only analysis.
