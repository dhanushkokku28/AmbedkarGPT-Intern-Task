#!/usr/bin/env python3
"""
AmbedkarGPT-Intern-Task - main.py

Command-line RAG Q&A using:
- LangChain
- ChromaDB (local, persisted)
- HuggingFace embeddings: sentence-transformers/all-MiniLM-L6-v2
- Ollama LLM with Mistral (local)

Requirements:
- Python 3.8+
- Run `pip install -r requirements.txt`
- Start Ollama (see README) before running this script

This script strictly answers questions only from the content in `speech.txt`.
"""

import os
import sys
from typing import List

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import Ollama


SPEECH_FILE = "speech.txt"
PERSIST_DIR = "chroma_db"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def load_speech_text(path: str) -> str:
    """Load the local speech text file and return its content."""
    if not os.path.exists(path):
        print(f"Error: '{path}' not found. Please ensure the file exists.")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into chunks suitable for embedding and retrieval."""
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
    )
    return splitter.split_text(text)


def create_or_load_vectorstore(chunks: List[str], persist_dir: str) -> Chroma:
    """Create a Chroma vectorstore from text chunks or load an existing persisted one.

    The embeddings use the local HuggingFace model `sentence-transformers/all-MiniLM-L6-v2`.
    """
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # If the persistence directory already has a Chroma DB, try loading it first.
    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        try:
            vectordb = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
            print(f"Loaded existing ChromaDB from '{persist_dir}'.")
            return vectordb
        except Exception as e:
            print("Warning: Failed to load existing Chroma DB, will recreate it.", e)

    # Otherwise, create the collection from the provided chunks and persist it.
    print("Creating ChromaDB from text chunks. This may take a moment...")
    vectordb = Chroma.from_texts(texts=chunks, embedding=embeddings, persist_directory=persist_dir)
    try:
        vectordb.persist()
    except Exception:
        # Some langchain/chroma versions persist automatically; ignore if not supported.
        pass
    print(f"ChromaDB created and persisted at '{persist_dir}'.")
    return vectordb


def build_retrieval_qa(vectordb: Chroma) -> RetrievalQA:
    """Build a RetrievalQA chain using Ollama Mistral as the local LLM.

    Note: Ensure `ollama serve` is running locally (see README).
    """
    # Create an Ollama LLM wrapper that talks to the local Ollama server.
    # The default Ollama server URL (when running `ollama serve`) is http://localhost:11434
    llm = Ollama(model="mistral", base_url="http://localhost:11434")

    # Use the vectorstore as a retriever. Adjust `k` for number of retrieved chunks.
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})

    # Build RetrievalQA chain. Use a simple chain_type like 'stuff'.
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
    return qa


def run_cli_loop(qa_chain: RetrievalQA):
    """Interactive CLI loop for continuous Q&A. Type 'exit' or 'quit' to stop."""
    print("\nAmbedkarGPT-Intern-Task — Q&A (answers only from speech.txt)")
    print("Type a question and press Enter. Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            query = input("Question> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        # Run the RetrievalQA chain. We get a dict with 'result' and 'source_documents'.
        try:
            res = qa_chain(query)
        except Exception as e:
            print("Error while querying the LLM/retriever:", e)
            continue

        answer = res.get("result") or res.get("answer") or str(res)
        print("\nAnswer:\n", answer.strip(), "\n")

        # Optionally print returned source snippets to show evidence
        docs = res.get("source_documents") or []
        if docs:
            print("Sources (excerpts retrieved):")
            for i, d in enumerate(docs, start=1):
                snippet = d.page_content.strip().replace("\n", " ")[:500]
                print(f" - [{i}] {snippet}...")
            print("")


def main():
    # Load speech text
    raw = load_speech_text(SPEECH_FILE)

    # Chunk text
    chunks = chunk_text(raw)
    print(f"Split speech into {len(chunks)} chunk(s).")

    # Create or load Chroma DB
    vectordb = create_or_load_vectorstore(chunks, PERSIST_DIR)

    # Build retrieval QA chain
    qa_chain = build_retrieval_qa(vectordb)

    # Run CLI loop
    run_cli_loop(qa_chain)


if __name__ == "__main__":
    main()
