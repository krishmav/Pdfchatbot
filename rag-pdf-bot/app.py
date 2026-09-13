"""
RAG PDF Bot
-----------
A multi-PDF Retrieval-Augmented Generation chatbot.

Stack: Streamlit (UI) + LangChain (orchestration) + FAISS (vector search)
       + Google Gemini (LLM + embeddings)

Run with:
    streamlit run app.py
"""

import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src.ingest import load_and_chunk_pdfs
from src.vectorstore import build_vectorstore, load_vectorstore, save_vectorstore
from src.rag_chain import build_rag_chain

load_dotenv()

APP_TITLE = "📄 RAG PDF Bot"
INDEX_DIR = "faiss_index"

st.set_page_config(page_title="RAG PDF Bot", page_icon="📄", layout="wide")


def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = None
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = []


def sidebar():
    with st.sidebar:
        st.header("⚙️ Setup")

        api_key = st.text_input(
            "Gemini API Key",
            type="password",
            value=os.getenv("GOOGLE_API_KEY", ""),
            help="Get a free key at https://aistudio.google.com/app/apikey",
        )
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key

        st.divider()
        st.subheader("📚 Upload PDFs")
        uploaded_files = st.file_uploader(
            "Upload one or more PDF files",
            type=["pdf"],
            accept_multiple_files=True,
        )

        col1, col2 = st.columns(2)
        with col1:
            chunk_size = st.number_input("Chunk size", 200, 4000, 1000, step=100)
        with col2:
            chunk_overlap = st.number_input("Chunk overlap", 0, 1000, 150, step=50)

        top_k = st.slider("Top-K chunks retrieved", 1, 10, 4)

        process_btn = st.button("🚀 Process Documents", use_container_width=True, type="primary")

        st.divider()
        if st.session_state.processed_files:
            st.subheader("✅ Indexed files")
            for f in st.session_state.processed_files:
                st.caption(f"• {f}")

        if st.button("🗑️ Clear session", use_container_width=True):
            st.session_state.messages = []
            st.session_state.vectorstore = None
            st.session_state.rag_chain = None
            st.session_state.processed_files = []
            st.rerun()

        return uploaded_files, process_btn, chunk_size, chunk_overlap, top_k, api_key


def process_documents(uploaded_files, chunk_size, chunk_overlap, top_k, api_key):
    if not api_key:
        st.error("Please enter your Gemini API key in the sidebar.")
        return
    if not uploaded_files:
        st.error("Please upload at least one PDF.")
        return

    with st.spinner("Reading and chunking PDFs..."):
        with tempfile.TemporaryDirectory() as tmp_dir:
            paths = []
            for uf in uploaded_files:
                p = Path(tmp_dir) / uf.name
                p.write_bytes(uf.getbuffer())
                paths.append(str(p))
            chunks = load_and_chunk_pdfs(paths, chunk_size, chunk_overlap)

    if not chunks:
        st.error("No extractable text found in the uploaded PDF(s).")
        return

    with st.spinner(f"Embedding {len(chunks)} chunks and building FAISS index..."):
        vectorstore = build_vectorstore(chunks)
        save_vectorstore(vectorstore, INDEX_DIR)

    with st.spinner("Wiring up the RAG chain..."):
        chain = build_rag_chain(vectorstore, top_k=top_k)

    st.session_state.vectorstore = vectorstore
    st.session_state.rag_chain = chain
    st.session_state.processed_files = [uf.name for uf in uploaded_files]
    st.session_state.messages = []
    st.success(f"Indexed {len(chunks)} chunks from {len(uploaded_files)} file(s). Ask away!")


def chat_area():
    st.title(APP_TITLE)
    st.caption("Upload PDFs, then ask natural-language questions grounded in their content.")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📎 Sources"):
                    for i, s in enumerate(msg["sources"], 1):
                        st.markdown(f"**[{i}] {s['source']} (chunk {s['chunk']})**")
                        st.caption(s["text"][:400] + ("..." if len(s["text"]) > 400 else ""))

    if prompt := st.chat_input("Ask a question about your documents..."):
        if st.session_state.rag_chain is None:
            st.warning("Upload and process at least one PDF first (see sidebar).")
            return

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            streamed_text = ""
            sources = []
            try:
                for chunk, chunk_sources in st.session_state.rag_chain.stream_answer(prompt):
                    streamed_text += chunk
                    placeholder.markdown(streamed_text + "▌")
                    if chunk_sources:
                        sources = chunk_sources
                placeholder.markdown(streamed_text)
                if sources:
                    with st.expander("📎 Sources"):
                        for i, s in enumerate(sources, 1):
                            st.markdown(f"**[{i}] {s['source']} (chunk {s['chunk']})**")
                            st.caption(s["text"][:400] + ("..." if len(s["text"]) > 400 else ""))
            except Exception as e:
                streamed_text = f"⚠️ Error generating answer: {e}"
                placeholder.markdown(streamed_text)

        st.session_state.messages.append(
            {"role": "assistant", "content": streamed_text, "sources": sources}
        )


def main():
    init_session_state()
    uploaded_files, process_btn, chunk_size, chunk_overlap, top_k, api_key = sidebar()

    if process_btn:
        process_documents(uploaded_files, chunk_size, chunk_overlap, top_k, api_key)

    chat_area()


if __name__ == "__main__":
    main()
