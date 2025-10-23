# src/utils.py
import os
from langchain.docstore.document import Document

def load_txt_files(data_dir="./data"):
    """
    Load all .txt files from the specified directory into Document objects.
    Each Document has page_content and metadata with source filename.
    """
    documents = []
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory '{data_dir}' does not exist.")
    
    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            path = os.path.join(data_dir, filename)
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
                documents.append(Document(page_content=content, metadata={"source": filename}))
    return documents

def ensure_dir(path):
    """
    Ensure the given directory exists. If not, create it.
    """
    os.makedirs(path, exist_ok=True)

def chunk_documents(documents, chunk_size=500, chunk_overlap=100):
    """
    Split documents into smaller chunks for better retrieval.
    Returns a list of Document objects with the same metadata.
    """
    from langchain.text_splitter import CharacterTextSplitter

    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    all_chunks = []
    for doc in documents:
        chunks = splitter.split_text(doc.page_content)
        for chunk in chunks:
            all_chunks.append(Document(page_content=chunk, metadata=doc.metadata))
    return all_chunks

def format_response(question, answer, docs):
    """
    Format the response dictionary with sources for API or CLI output.
    """
    return {
        "question": question,
        "answer": answer,
        "sources": [doc.metadata.get("source") for doc in docs]
    }
