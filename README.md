RAG FAISS Demo

A Retrieval-Augmented Generation (RAG) pipeline using LangChain, FAISS, and Azure OpenAI to answer questions from local documents. This project supports a FastAPI endpoint and a CLI interface for querying.

Features

Load local text documents and split them into chunks.

Generate embeddings using Azure OpenAI Embeddings.

Build a FAISS vector store for fast similarity search.

RAG pipeline that combines retriever + LLM to answer questions.

FastAPI API for programmatic access.

CLI interface for quick testing.

Requirements

Python 3.11+

Pip packages:

pip install -r requirements.txt


requirements.txt should include (example):

fastapi
uvicorn
python-dotenv
langchain
langchain-community
faiss-cpu


Azure OpenAI account with text-embedding-ada-002 and gpt-4o (or your preferred deployment).

Environment variables in .env:

AZURE_OPENAI_ENDPOINT=your_endpoint_here
AZURE_OPENAI_API_KEY=your_api_key_here
OPENAI_API_VERSION=2025-01-01-preview

Setup

Clone the repository:

git clone <repo-url>
cd rag-faiss-demo


Create and activate virtual environment:

python -m venv venv
source venv/bin/activate      # Linux/macOS
venv\Scripts\activate         # Windows


Install dependencies:

pip install -r requirements.txt


Add your .env file with Azure OpenAI credentials.

Place your .txt documents in DATA_DIR (default: ./data).

Running the Pipeline
1. Ingest Data
python src/ingest_data.py


This splits documents into chunks, generates embeddings, and builds a FAISS index in INDEX_DIR.

2. Run CLI
python src/query_rag.py


Ask questions interactively.

Use exit or quit to close.

3. Run API
uvicorn src.api:app --reload


Open Swagger UI: http://127.0.0.1:8000/docs

POST /query with JSON:

{
  "question": "What is a risk flag?\nHow is it identified?"
}


Use \n for multiple lines in the question field.