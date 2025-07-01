import streamlit as st
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyMuPDFLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import os
import tempfile
import shutil
import time
import requests
from urllib.parse import urlparse
import re

# 1. Optimized Models for Speed
@st.cache_resource
def initialize_models():
    """Initialize LLM and embeddings optimized for speed."""
    llm = ChatOllama(
        model="tinyllama",    # Fastest model - only 637MB!
        temperature=0.1,
        num_ctx=1024,         # Reduced context for speed
        top_p=0.9,
        repeat_penalty=1.1,
        # Speed optimizations
        num_thread=8,         # multiple threads
        num_predict=256,      # Limit response length for speed
        stop=["Human:", "Assistant:", "\n\n"]  # Stop tokens
    )
    
    # Fastest embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={
            'normalize_embeddings': False,  # Skiped normalization for speed
            'batch_size': 32
        }
    )
    return llm, embeddings

# 2. URL validation and PDF detection
def is_valid_pdf_url(url):
    """Check if URL is valid and points to a PDF."""
    try:
        # Basic URL validation
        parsed = urlparse(url)
        if not all([parsed.scheme, parsed.netloc]):
            return False, "Invalid URL format"
        
        # Check if URL ends with .pdf
        if url.lower().endswith('.pdf'):
            return True, "Direct PDF URL detected"
        
        # Check HTTP headers for PDF content
        try:
            response = requests.head(url, timeout=10, allow_redirects=True)
            content_type = response.headers.get('content-type', '').lower()
            
            if 'application/pdf' in content_type:
                return True, "PDF content detected"
            elif 'text/html' in content_type:
                return False, "HTML page detected - please provide direct PDF link"
            else:
                return False, f"Unknown content type: {content_type}"
        except requests.RequestException:
            # If head request fails, assume it might be a PDF and let WebBaseLoader handle it
            return True, "URL validation skipped - will attempt to load"
            
    except Exception as e:
        return False, f"URL validation error: {str(e)}"

# 3. Enhanced PDF processing with URL support
def process_and_vectorize_pdf(pdf_source, source_type="file", persist_directory="faiss_db_deconstructor"):
    """Lightning fast PDF processing with FAISS vector store - supports both files and URLs."""
    
    # Clear existing vector store directory
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)
    
    documents = []
    
    try:
        if source_type == "file":
            # PDF file processing
            loader = PyMuPDFLoader(pdf_source)
            documents = loader.load()
        
        elif source_type == "url":
            # URL processing with WebBaseLoader
            st.info("Loading PDF from URL...")
            
            # First validate URL
            is_valid, message = is_valid_pdf_url(pdf_source)
            if not is_valid:
                st.error(f"❌ {message}")
                return None
            
            # Configure WebBaseLoader for PDF
            loader = WebBaseLoader(
                web_paths=[pdf_source],
                header_template={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                },
                requests_kwargs={'timeout': 30}
            )
            
            documents = loader.load()
            
            # Additional processing for web-loaded content
            if documents:
                for doc in documents:
                    # Clean up common web artifacts
                    doc.page_content = re.sub(r'<[^>]+>', '', doc.page_content)  # Remove HTML tags
                    doc.page_content = re.sub(r'\s+', ' ', doc.page_content)    # Normalize whitespace
                    doc.page_content = doc.page_content.strip()
        
        if not documents:
            st.error("Could not extract text from the PDF. Please ensure it's a valid PDF file/URL.")
            return None
        
        # Filter out empty documents
        documents = [doc for doc in documents if doc.page_content.strip()]
        
        if not documents:
            st.error("No text content found in the PDF.")
            return None
        
        # Aggressive text splitting for speed
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,    # Smaller chunks for faster processing
            chunk_overlap=40,  # Minimal overlap for speed
            separators=["\n\n", "\n", ". "],
            length_function=len,
        )
        
        splits = text_splitter.split_documents(documents)
        
        splits = [split for split in splits if len(split.page_content.strip()) > 30]
        
        # Limit to first 150 chunks for maximum speed
        if len(splits) > 150:
            splits = splits[:150]
            st.info(f"Processing first 150 chunks out of {len(splits)} total for optimal speed")
        
        if not splits:
            st.error("No meaningful text chunks could be created from the PDF.")
            return None
        
        _, embeddings = initialize_models()
        
        # Used FAISS instead of Chroma - much faster!
        vectorstore = FAISS.from_documents(
            documents=splits, 
            embedding=embeddings
        )
        
        # Save FAISS index for persistence (optional)
        try:
            os.makedirs(persist_directory, exist_ok=True)
            vectorstore.save_local(persist_directory)
        except:
            pass  # Skip saving if there's an issue
        
        st.success(f"Processing complete! Created {len(splits)} chunks.")
        return vectorstore
        
    except Exception as e:
        st.error(f"❌ Error processing PDF: {str(e)}")
        return None

# 4. Simplified, fast prompt template
def create_custom_prompt():
    """Create a fast, concise prompt template."""
    
    custom_prompt = PromptTemplate(
        template="""Use this research paper context to answer the question briefly and accurately:

Context: {context}

Chat History: {chat_history}

Question: {question}

Answer (be specific and concise):""",
        input_variables=["context", "chat_history", "question"]
    )
    
    return custom_prompt

# 5. Speed-optimized conversation chain
def create_conversation_chain(vectorstore):
    """Create a lightning-fast conversational chain."""
    
    llm, _ = initialize_models()
    
    # Fast retriever - fewer documents, similarity search
    retriever = vectorstore.as_retriever(
        search_type="similarity",  # Much faster than MMR
        search_kwargs={
            "k": 3,  # Fewer chunks for speed
        }
    )
    
    # Lightweight memory with limits
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True,
        output_key='answer',
        max_token_limit=500  # Limit memory for speed
    )
    
    # Create optimized chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=False,  # Skip source docs for speed
        verbose=False,  # Turn off verbose logging
        combine_docs_chain_kwargs={"prompt": create_custom_prompt()}
    )
    
    return qa_chain

st.set_page_config(
    page_title="Research Paper Deconstructor", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Research Paper Deconstructor")
st.markdown("*Powered by TinyLlama - Open Source RAG*")

# Initialize session state
if "conversation_chain" not in st.session_state:
    st.session_state.conversation_chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False
if "current_source" not in st.session_state:
    st.session_state.current_source = None

# Enhanced Sidebar
with st.sidebar:
    st.header("Upload Research Paper")
    st.markdown("Upload a PDF file or provide a direct PDF URL")
    
    # Input method selection
    input_method = st.radio(
        "Choose input method:",
        ["Upload PDF File", "PDF URL"],
        horizontal=True
    )
    
    uploaded_file = None
    pdf_url = None
    
    if input_method == "Upload PDF File":
        uploaded_file = st.file_uploader(
            "Choose a PDF file", 
            type="pdf",
            help="Upload academic papers, research documents, or technical reports"
        )
        
        if uploaded_file is not None:
            st.info(f"**File:** {uploaded_file.name}")
            st.info(f"**Size:** {uploaded_file.size / 1024:.1f} KB")
    
    else:  # PDF URL
        pdf_url = st.text_input(
            "Enter direct PDF URL:",
            placeholder="https://example.com/paper.pdf",
            help="Provide a direct link to a PDF file (must end with .pdf or serve PDF content)"
        )
        
        if pdf_url:
            st.info(f"**URL:** {pdf_url}")
            
            # Real-time URL validation
            is_valid, message = is_valid_pdf_url(pdf_url)
            if is_valid:
                st.success(f"{message}")
            else:
                st.warning(f"⚠️ {message}")
    
    process_button = st.button("Process PDF", type="primary")
    
    if process_button:
        if input_method == "Upload PDF File" and uploaded_file is not None:
            with st.spinner("Processing PDF file... This may take a few moments."):
                # Create temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    temp_file_path = tmp_file.name
                
                try:
                    # Process the PDF file
                    vector_store = process_and_vectorize_pdf(temp_file_path, source_type="file")
                    
                    if vector_store is not None:
                        st.session_state.vector_store = vector_store
                        st.session_state.conversation_chain = create_conversation_chain(vector_store)
                        st.session_state.pdf_processed = True
                        st.session_state.current_source = uploaded_file.name
                        st.session_state.chat_history = []  # Reset chat history
                        st.rerun()
                
                except Exception as e:
                    st.error(f"Error processing PDF: {str(e)}")
                
                finally:
                    # Clean up temporary file
                    if os.path.exists(temp_file_path):
                        os.unlink(temp_file_path)
        
        elif input_method == "PDF URL" and pdf_url:
            with st.spinner("Processing PDF from URL... This may take a few moments."):
                try:
                    # Process the PDF URL
                    vector_store = process_and_vectorize_pdf(pdf_url, source_type="url")
                    
                    if vector_store is not None:
                        st.session_state.vector_store = vector_store
                        st.session_state.conversation_chain = create_conversation_chain(vector_store)
                        st.session_state.pdf_processed = True
                        st.session_state.current_source = pdf_url
                        st.session_state.chat_history = []  # Reset chat history
                        st.rerun()
                
                except Exception as e:
                    st.error(f"Error processing PDF from URL: {str(e)}")
        
        else:
            st.warning("⚠️ Please provide a PDF file or valid PDF URL first.")
    
    # Display current source
    if st.session_state.pdf_processed and st.session_state.current_source:
        st.divider()
        st.markdown("**Current Source:**")
        if st.session_state.current_source.startswith("http"):
            st.markdown(f"URL: {st.session_state.current_source}")
        else:
            st.markdown(f"File: {st.session_state.current_source}")
    
    # Add helpful tips
    with st.expander("Lightning Fast Results"):
        st.markdown("""
        **Expected Performance:**
        - PDF File Processing: 5-15 seconds
        - PDF URL Processing: 10-30 seconds
        - Question Response: 1-3 seconds
        
        **Supported PDF Sources:**
        - Local PDF files
        - Direct PDF URLs (ending with .pdf)
        - URLs serving PDF content
        
        **For Best Results:**
        - Use direct PDF links (not webpage links)
        - Ensure PDFs are text-based (not scanned images)
        - Ask specific, direct questions
        - Academic papers work best
        
        **Example PDF URLs:**
        - ArXiv: https://arxiv.org/pdf/xxxx.xxxx.pdf
        - Research Gate: Direct PDF links
        - University repositories: Direct PDF links
        """)

# Main Chat Interface
col1, col2 = st.columns([3, 1])

with col1:
    st.header("Chat with Your Document")

with col2:
    if st.session_state.pdf_processed:
        st.success("Document is Ready")
    else:
        st.warning("No PDF Processed")

if not st.session_state.pdf_processed:
    st.info("Please upload a PDF file or provide a PDF URL using the sidebar to begin chatting with your document.")
    
    # Sample questions
    st.subheader("Example Questions You Can Ask:")
    example_questions = [
        "What is the main research question or hypothesis?",
        "What methodology was used in this study?",
        "What are the key findings and results?",
        "What are the limitations mentioned by the authors?",
        "How does this work compare to previous research?",
        "What future work do the authors suggest?"
    ]
    
    for i, question in enumerate(example_questions, 1):
        st.markdown(f"**{i}.** {question}")

else:
    # Display chat messages
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about the research paper..."):
        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response with speed optimizations and timing
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            try:
                # Show immediate feedback with timer
                start_time = time.time()
                message_placeholder.markdown("Generating response...")
                
                # Fast response generation
                response = st.session_state.conversation_chain({"question": prompt})
                bot_response = response['answer']
                
                # Calculate response time
                end_time = time.time()
                response_time = end_time - start_time
                
                # Display response with timing
                final_response = f"{bot_response}\n\n*Response time: {response_time:.1f}s*"
                message_placeholder.markdown(final_response)
                
                # Add to chat history
                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "content": final_response
                })
                
            except Exception as e:
                message_placeholder.error(f"❌ Error: {str(e)}")
                st.info("Try a simpler question or check if TinyLlama is properly installed.")

# Clear chat button
if st.session_state.pdf_processed:
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
    
    with col2:
        if st.button("Reset PDF"):
            st.session_state.conversation_chain = None
            st.session_state.chat_history = []
            st.session_state.vector_store = None
            st.session_state.pdf_processed = False
            st.session_state.current_source = None
            st.rerun()