# ResearchPaper-Assistant

**An AI-powered research assistant that constructs IEEE-formatted papers from GitHub repositories and deconstructs existing papers for interactive Q&A.**

This project is a comprehensive, dual-function application designed to accelerate academic and technical writing workflows. It leverages state-of-the-art Large Language Models (LLMs) through both high-speed cloud APIs (Groq) and private, local inference (Ollama), all orchestrated by the LangChain framework.

1.  **The Constructor**: Analyzes an entire GitHub repository—its code, structure, and documentation—to automatically generate a well-structured, multi-page research paper in the standard IEEE format. It uses a sophisticated RAG pipeline to ground the generated content in the actual codebase.

2.  **The Deconstructor**: Provides a fast, interactive chat interface to "deconstruct" any existing research paper. Users can upload a PDF or provide a URL to start a conversation, asking complex questions and receiving instant, context-aware answers from a locally-run LLM.

Tech Stack
1. Frontend: Streamlit
2. LLM Orchestration: LangChain
3. Cloud LLM Service: Groq (Llama-3.1-8B-Instant, Llama-3.3-70B-Versatile)
4. Vector Databases: Constructor - FAISS , Deconstructor - ChromaDB
5. Database: SQLite/Postgres (for session & chat history)
6. Embeddings: Hugging Face Sentence Transformers (all-MiniLM-L6-v2)
7. PDF Generation: ReportLab
8. PDF Parsing: PyMuPDF, PyPDF2
9. API/Web Interaction: requests

Setup and Installation 
1. Clone the repository
2. Create and activate a virtual environment
3. Install dependencies
4. Set up environment variables - Add your GROQ_API_KEY
5. Prepare the local LLM - Ollama ( run ollama pull tinyollama in your terminal)
6. Run the Streamlit App - streamlit run Home.py

