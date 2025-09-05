from mcp.server.fastmcp.server import FastMCP
import logging

# Import the necessary components from your RAG bot's main script
from main import (
    ingest_docs,
    chunk_docs,
    rebuild_vectorstore,
    load_hf_llm,
)

# --- Config (from your main.py) ---
CSV_PATH = "data/ToTo.csv"
PDF_PATH = "data/test.pdf"
TEXT_PATH = "data/sg60-national-day.txt"
TOP_K = 3

# Use FastMCP instead of FastAPI
mcp = FastMCP("mcp_rag_server")

# --- Global RAG components initialization ---
# This part runs only once when the server starts
logging.info("Initializing RAG bot components...")
try:
    # 1. Ingest documents
    docs = ingest_docs(CSV_PATH, PDF_PATH, TEXT_PATH)
    logging.info(f"Total documents loaded: {len(docs)}")

    # 2. Chunk documents
    chunks = chunk_docs(docs)
    logging.info(f"Total chunks created: {len(chunks)}")

    # 3. Build/rebuild the vector store
    # Using the rebuild function to ensure a fresh start
    vectordb = rebuild_vectorstore(chunks)
    retriever = vectordb.as_retriever()
    logging.info("Vector store initialized and ready.")

    # 4. Load the HuggingFace LLM
    llm = load_hf_llm("google/flan-t5-large")
    logging.info("HuggingFace LLM loaded.")

    is_ready = True
    logging.info("RAG Bot server is ready to handle queries.")
except Exception as e:
    is_ready = False
    logging.info(f"Error during initialization: {e}")

@mcp.tool()
async def rag_query_endpoint(query: str):
    """
    This tool receives a query, performs a RAG process, and returns a RAG-based answer.
    """
    # if not is_ready:
    #     raise HTTPException(status_code=503, detail="Server is not ready. RAG components failed to initialize.")

    logging.info(f"Received RAG query via tool call: '{query}'")

    try:
        # Use the pre-initialized components to get the answer
        docs = retriever.invoke(query)[:TOP_K]
        context = "\n\n---\n\n".join([d.page_content for d in docs]) or "No context found."
        prompt = f"Answer the question using ONLY the context below:\n\nCONTEXT:\n{context}\n\nQUESTION: {query}"
        answer = llm.invoke(prompt)

        return {
            "query": query,
            "answer": answer,
            "source_documents": [{"page_content": d.page_content, "metadata": d.metadata} for d in docs]
        }
    except Exception as e:
        logging.info(f"Error processing RAG query: {e}")
        # raise HTTPException(status_code=500, detail="An error occurred while processing your query.")

@mcp.tool()
async def search_endpoint(query: str):
    """
    This tool performs a search query and returns relevant documents from the vector store.
    """
    # if not is_ready:
    #     raise HTTPException(status_code=503, detail="Server is not ready. RAG components failed to initialize.")

    logging.info(f"Received search tool query: '{query}'")

    try:
        # Use the pre-initialized retriever to find relevant documents
        docs = retriever.invoke(query)[:TOP_K]

        return {
            "query": query,
            "source_documents": [{"page_content": d.page_content, "metadata": d.metadata} for d in docs]
        }
    except Exception as e:
        logging.info(f"Error processing search query: {e}")
        # raise HTTPException(status_code=500, detail="An error occurred while processing your search query.")

@mcp.tool()
async def health_check():
    return {"status": "ok", "ready": is_ready}

if __name__ == "__main__":
    # uvicorn.run(mcp, host="0.0.0.0")
    logging.info("RAG MCP Server is running...")
    mcp.run(transport="stdio")