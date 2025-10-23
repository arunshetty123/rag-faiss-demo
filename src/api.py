from fastapi import FastAPI
from pydantic import BaseModel, Field
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from src.config import USE_AZURE, LLM_DEPLOYMENT, INDEX_DIR
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="RAG FAISS API",
    description="RAG pipeline powered by LangChain + FAISS",
    version="1.0"
)

vectorstore = None
retriever = None
llm = None


# ---- Pydantic model for request ----
class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        title="Your Question",
        description="Enter your question here. Multiline input is allowed.",
        example="Line 1\nLine 2\nLine 3",
        max_length=2000
    )


# ---- Helper to load Azure/OpenAI LLM ----
def get_llm():
    if USE_AZURE:
        return AzureChatOpenAI(
            azure_deployment=LLM_DEPLOYMENT,
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("OPENAI_API_VERSION", "2024-02-01"),
            temperature=0
        )
    else:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-4o", temperature=0)


# ---- load FAISS and models ----
@app.on_event("startup")
def load_resources():
    global vectorstore, retriever, llm

    if not os.path.exists(INDEX_DIR):
        raise RuntimeError(f"FAISS index not found at {INDEX_DIR}. Run ingest_data.py first.")

    # Initialize embeddings
    if USE_AZURE:
        embeddings = AzureOpenAIEmbeddings(
            model="text-embedding-ada-002",
            azure_deployment="text-embedding-ada-002",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("OPENAI_API_VERSION", "2024-02-01")
        )
    else:
        from langchain_openai import OpenAIEmbeddings
        embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

    # Load FAISS index
    vectorstore = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = get_llm()

    print("RAG API is ready and FAISS index loaded.")


# ---- API Endpoint ----
@app.post("/query")
def query_rag(request: QueryRequest):
    question = request.question
    docs = retriever.invoke(question)
    context = "\n---\n".join([d.page_content for d in docs])

    prompt = PromptTemplate(
        template=(
            "You are a helpful assistant. Use the provided context to answer accurately.\n\n"
            "Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
        ),
        input_variables=["context", "question"]
    )

    final_prompt = prompt.format(context=context, question=question)
    response = llm.invoke(final_prompt)

    return {
        "question": question,
        "answer": response.content if hasattr(response, "content") else str(response),
        "sources": [d.metadata.get("source") for d in docs]
    }


@app.post("/ask-anything")
def askanything(request: QueryRequest):
    question = request.question

    prompt = PromptTemplate(
        template=(
            "You are a helpful assistant. Answer the user question accurately in detail\n\n"
            "Question:\n{question}\n\nAnswer:"
        ),
        input_variables=["question"]
    )

    final_prompt = prompt.format(question=question)
    response = llm.invoke(final_prompt)

    return {
        "question": question,
        "answer": response.content if hasattr(response, "content") else str(response)
    }


# ---- Root Endpoint ----
@app.get("/")
def home():
    return {"message": "Welcome to the RAG FAISS API. Use POST /query with a question."}
