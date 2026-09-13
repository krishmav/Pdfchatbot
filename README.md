# 📄 RAG PDF Bot

A multi-PDF Retrieval-Augmented Generation (RAG) chatbot that lets you
ask natural-language questions about the content of one or more PDF
documents, with context-aware, streamed answers.

**Stack:** Python · Streamlit · LangChain · FAISS · Gemini API

## Features

- Upload and query **multiple PDFs** at once
- Automatic **document chunking** (configurable size/overlap) via LangChain's
  `RecursiveCharacterTextSplitter`
- **Embeddings + FAISS similarity search** for fast, relevant context retrieval
- **Gemini LLM** integration with a prompt template that grounds answers in
  retrieved context
- **Streaming responses** token-by-token in the chat UI
- Source citations (file name + chunk) shown under every answer
- Persisted FAISS index (`faiss_index/`) so you don't have to re-embed on
  every run

## Project structure

```
rag-pdf-bot/
├── app.py                 # Streamlit UI and app entrypoint
├── src/
│   ├── ingest.py           # PDF loading + chunking
│   ├── vectorstore.py      # Embeddings + FAISS build/save/load
│   └── rag_chain.py        # Prompt template + Gemini streaming chain
├── sample_docs/            # Drop sample PDFs here for testing
├── .streamlit/config.toml  # UI theme
├── requirements.txt
├── .env.example
└── README.md
```

## Setup

1. **Clone / unzip** the project and move into it:
   ```bash
   cd rag-pdf-bot
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Get a Gemini API key** for free at
   [Google AI Studio](https://aistudio.google.com/app/apikey).

5. **Configure your key** — either:
   - Copy `.env.example` to `.env` and paste your key, or
   - Paste it directly into the sidebar when the app runs.

6. **Run the app:**
   ```bash
   streamlit run app.py
   ```

7. Open the local URL Streamlit prints (usually `http://localhost:8501`),
   upload PDFs in the sidebar, click **Process Documents**, and start
   chatting.

## How it works

1. **Ingest** — Uploaded PDFs are parsed with `PyPDFLoader` and split into
   overlapping chunks (default: 1000 chars, 150 overlap) so relevant
   passages aren't cut off mid-thought.
2. **Embed & Index** — Each chunk is embedded with Gemini's
   `embedding-001` model and stored in a FAISS vector index for fast
   nearest-neighbor search.
3. **Retrieve** — When you ask a question, the top-K most similar chunks
   are retrieved from FAISS.
4. **Generate** — The retrieved chunks are inserted into a prompt template
   along with your question and recent chat history, then sent to
   `gemini-3.5-flash`, which streams its answer back token-by-token.

## Configuration options (sidebar)

| Setting        | Description                                  | Default |
|----------------|-----------------------------------------------|---------|
| Chunk size     | Characters per chunk                          | 1000    |
| Chunk overlap  | Overlap between consecutive chunks            | 150     |
| Top-K          | Number of chunks retrieved per question       | 4       |
