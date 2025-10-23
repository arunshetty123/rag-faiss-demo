import os
import time
from langchain_openai import AzureOpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from config import INDEX_DIR, DATA_DIR
from dotenv import load_dotenv

load_dotenv()

# Set Azure environment variables
os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv("AZURE_OPENAI_ENDPOINT")
os.environ["AZURE_OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")
os.environ["OPENAI_API_VERSION"] = os.getenv("OPENAI_API_VERSION", "2025-01-01-preview")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
BATCH_SIZE = 50       # Number of chunks to process in one batch
WAIT_TIME = 60        # Seconds to wait before processing next batch

def load_documents():
    """Load all text files from the data directory."""
    documents = []
    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".txt"):
            path = os.path.join(DATA_DIR, filename)
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
                documents.append(Document(page_content=content, metadata={"source": filename}))
    return documents

def chunk_documents(documents):
    """Split documents into smaller chunks."""
    splitter = CharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    all_chunks = []
    for doc in documents:
        chunks = splitter.split_text(doc.page_content)
        for chunk in chunks:
            all_chunks.append(Document(page_content=chunk, metadata=doc.metadata))
    return all_chunks

def main():
    os.makedirs(INDEX_DIR, exist_ok=True)

    print("Loading documents...")
    docs = load_documents()
    if not docs:
        print("No documents found in", DATA_DIR)
        return

    print("Splitting documents into chunks...")
    docs_chunks = chunk_documents(docs)
    print(f"Total chunks created: {len(docs_chunks)}")

    print("Initializing Azure OpenAI embeddings...")
    embeddings = AzureOpenAIEmbeddings(model="text-embedding-ada-002")

    vectorstore = None

    # Process in batches to avoid rate limit issues
    for i in range(0, len(docs_chunks), BATCH_SIZE):
        batch = docs_chunks[i:i + BATCH_SIZE]
        print(f"Processing batch {i // BATCH_SIZE + 1} ({len(batch)} chunks)...")

        if vectorstore is None:
            vectorstore = FAISS.from_documents(batch, embeddings)
        else:
            new_store = FAISS.from_documents(batch, embeddings)
            vectorstore.merge_from(new_store)

        # Wait between batches if more remain
        if i + BATCH_SIZE < len(docs_chunks):
            print(f"Waiting {WAIT_TIME} seconds before next batch...")
            time.sleep(WAIT_TIME)
            if i // BATCH_SIZE == 9:
                print(f"Waiting {WAIT_TIME} seconds more before next batch of 10 chunks...")
                time.sleep(WAIT_TIME)

    print(f"Saving FAISS index to {INDEX_DIR}...")
    vectorstore.save_local(INDEX_DIR)
    print("Index built successfully!")

if __name__ == "__main__":
    main()
