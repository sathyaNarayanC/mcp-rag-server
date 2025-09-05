
# 🚀 MCP RAG Server for Claude Desktop

A Retrieval-Augmented Generation (RAG) server that seamlessly integrates with Claude Desktop through the Model Context Protocol (MCP). This project enhances Claude's capabilities by providing access to your local document knowledge base through an advanced RAG pipeline.

## ✨ Features

### 📚 Document Processing
- Multi-format support (CSV, PDF, TXT)
- Smart document chunking with context preservation
- Efficient vector embeddings with Chroma DB
- Automated document indexing and updates

## 🛠️ Quick Start

### Prerequisites
- Python 3.9+
- [uv package manager](https://github.com/astral-sh/uv)
- 16GB+ RAM recommended
- Optional: CUDA-compatible GPU

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/mcp-rag-server.git
   cd mcp-rag-server
   ```

2. **Create and activate virtual environment**
   ```bash
   uv venv
   source .venv/bin/activate  # Unix/macOS
   # or
   .venv\Scripts\activate     # Windows
   ```

3. **Install dependencies**
   ```bash
   uv pip install -e .  # Install from pyproject.toml
   # or
   uv pip sync         # Install from lock file
   ```


### Configuration

1. **Prepare your documents**
   ```bash
   mkdir -p data/{csv,pdf,text}
   # Add your documents to respective folders
   ```

2. **Start the server**
   ```bash
   python mcp_rag_server.py
   ```

3. **Configure Claude Desktop**
   - Open Claude Desktop Settings
   - Navigate to Integrations
   - Add new MCP Server:
     ```
     Name: Local RAG Server
     URL: http://localhost:8000
     ```

## 🚀 Features & Performance

### High-Performance Processing
- Fast document indexing (~100 pages/minute)
- Efficient vector storage with Chroma
- GPU acceleration support
- Optimized memory usage (~100MB per 1000 pages)

### Enhanced Security
- 100% local processing
- No external API dependencies
- Secure document handling
- Configurable access controls

## 💻 Development

### Setup Development Environment
```bash
# Install dev dependencies
uv pip install pytest black flake8 isort mypy

# Update lock file
uv pip compile pyproject.toml -o uv.lock

# Sync project dependencies
uv pip sync
```

### Code Quality
```bash
# Format code
black .

# Run linters
flake8 .
mypy .

# Run tests
pytest tests/
```

## ⚙️ Advanced Configuration

### Document Processing
```python
# config.py
DOCUMENT_CONFIG = {
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "embeddings_model": "sentence-transformers/all-mpnet-base-v2"
}
```

### Retrieval Settings
```python
RETRIEVER_CONFIG = {
    "search_type": "similarity",  # or "mmr" for diversity
    "top_k": 3,
    "score_threshold": 0.7
}
```

### Custom Model Integration
```python
from langchain_huggingface import HuggingFaceEmbeddings

custom_embeddings = HuggingFaceEmbeddings(
    model_name="your-custom-model",
    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
)

7. Maintenance and Monitoring

Vector store management
Dependency updates
Performance monitoring
System health checks

8. Project Structure

.
├── mcp_rag_server.py    # MCP server
├── main.py              # Core implementation
├── pyproject.toml       # Dependencies
├── uv.lock             # Lock file
├── data/               # Documents
└── chroma_db/         # Vector store


