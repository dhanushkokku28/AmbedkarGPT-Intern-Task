AmbedkarGPT-Intern-Task
Local RAG-based CLI Q&A system that answers questions only from speech.txt.

Stack:

Python 3.8+
LangChain
ChromaDB (local persistent store)
HuggingFace embeddings: sentence-transformers/all-MiniLM-L6-v2
Ollama (local) with the mistral model
Setup
Create and activate a Python environment (optional but recommended).

PowerShell example:

python -m venv .venv
.\.venv\Scripts\Activate.ps1
Install Python dependencies:
pip install -r requirements.txt
Install and run Ollama locally (no API keys required):
Follow official instructions: https://ollama.com/docs/installation
Pull the Mistral model (if a pull command is available for your Ollama installation):
ollama pull mistral
Start the Ollama server (this exposes a local API that the script uses):
ollama serve
Note: Ollama typically serves on http://localhost:11434 which is what main.py uses.

(Optional) Replace speech.txt with the full text you are permitted to use.
Run
In the project folder run:

python main.py
Example terminal session:

AmbedkarGPT-Intern-Task — Q&A (answers only from speech.txt)
Type a question and press Enter. Type 'exit' or 'quit' to stop.

Question> What remedy does Ambedkar propose?

Answer:
 ... (LLM answer based only on speech.txt) ...

Sources (excerpts retrieved):
 - [1] The real remedy is to destroy the belief in the sanctity of the shastras...

Question> exit
Goodbye.
Notes & Troubleshooting
This project uses only local models and stores — no cloud API keys required.
If you get errors connecting to Ollama, confirm ollama serve is running and reachable at http://localhost:11434.
If Chroma DB creation fails, ensure you have write permissions in the project folder; the DB is saved in chroma_db/.
To update the text corpus, edit speech.txt and delete the chroma_db/ folder to force reindexing on next run.
If you'd like, I can also commit these files into your git repo and run a quick smoke test (if you want me to, say so).
