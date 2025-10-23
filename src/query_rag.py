# src/query_rag.py
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI, ChatOpenAI, AzureOpenAIEmbeddings
from langchain.schema.output_parser import StrOutputParser
from config import USE_AZURE, LLM_DEPLOYMENT, INDEX_DIR

load_dotenv()

# --- Azure environment setup ---
os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv("AZURE_OPENAI_ENDPOINT")
os.environ["AZURE_OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")
os.environ["OPENAI_API_VERSION"] = os.getenv("OPENAI_API_VERSION", "2025-01-01-preview")


def get_llm():
    """Initialize Azure or OpenAI LLM."""
    if USE_AZURE:
        return AzureChatOpenAI(
            azure_deployment=LLM_DEPLOYMENT,
            temperature=0
        )
    else:
        return ChatOpenAI(model_name="gpt-4o", temperature=0)


def load_index():
    """Load FAISS index and create retriever."""
    if not os.path.exists(INDEX_DIR):
        raise RuntimeError(f"FAISS index not found at {INDEX_DIR}. Run ingest_data.py first.")

    embeddings = (
        AzureOpenAIEmbeddings(model="text-embedding-ada-002")
        if USE_AZURE else None
    )

    vectorstore = FAISS.load_local(
        INDEX_DIR,
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorstore.as_retriever(search_kwargs={"k": 5})


def build_rag_chain(llm, retriever):
    """Create a LangChain Expression Language (LCEL) pipeline for RAG."""
    prompt = PromptTemplate(
        template="""You are a helpful assistant. Use the provided context to answer clearly.
Context:
{context}

Question:
{question}
""",
        input_variables=["context", "question"]
    )

    # LCEL pipeline — retrieval → context merge → prompt → LLM → output parse
    rag_chain = (
        {
            "context": retriever | (lambda docs: "\n---\n".join([d.page_content for d in docs])),
            "question": lambda x: x,
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


def main():
    """Simple CLI for RAG Q&A."""
    print("Loading FAISS index...")
    retriever = load_index()
    llm = get_llm()

    rag_chain = build_rag_chain(llm, retriever)

    print("RAG CLI ready. Type your questions below (or 'exit' to quit).")

    while True:
        question = input("\nEnter your question: ")
        if question.lower() in ["exit", "quit"]:
            break

        answer = rag_chain.invoke(question)
        print("\nAnswer:", answer)


if __name__ == "__main__":
    main()
