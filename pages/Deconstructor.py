import os
import tempfile
import time
import requests
import uuid
from typing import List, Dict, Optional
from datetime import datetime

import streamlit as st
from groq import Groq

# LangChain
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage, SystemMessage

# DB
from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey, create_engine, JSON
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

# PDF fallback
import PyPDF2

# ---------------------------
# Config
# ---------------------------
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_persist")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./deconstructor_sessions.db")

# ---------------------------
# Groq Client
# ---------------------------
def initialize_groq_client():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return None
    return Groq(api_key=api_key)

def call_groq_llm(prompt, max_tokens=1500):
    client = initialize_groq_client()
    if not client:
        return "Error: Groq API key missing."
    try:
        resp = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.2,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"Error generating response: {e}"

# ---------------------------
# DB Models
# ---------------------------
Base = declarative_base()
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {})
DBSession = sessionmaker(bind=engine)

class SessionRecord(Base):
    __tablename__ = "sessions"
    id = Column(String, primary_key=True)
    name = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_active = Column(DateTime, default=datetime.utcnow)
    messages = relationship("MessageRecord", back_populates="session", cascade="all, delete-orphan")
    documents = relationship("DocumentRecord", back_populates="session", cascade="all, delete-orphan")

class MessageRecord(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, ForeignKey("sessions.id"), index=True)
    role = Column(String)
    content = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    session = relationship("SessionRecord", back_populates="messages")

class DocumentRecord(Base):
    __tablename__ = "documents"
    id = Column(String, primary_key=True)
    session_id = Column(String, ForeignKey("sessions.id"), index=True)
    filename = Column(String)
    source = Column(String)
    doc_metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    session = relationship("SessionRecord", back_populates="documents")

Base.metadata.create_all(bind=engine)

# ---------------------------
# Embeddings + Chroma
# ---------------------------
@st.cache_resource
def initialize_embedding_model():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

@st.cache_resource
def init_chroma():
    return Chroma(
        collection_name="deconstructor_collection",
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=initialize_embedding_model()
    )

# ---------------------------
# PDF Utils
# ---------------------------
def pdf_to_text(path: str) -> str:
    try:
        loader = PyMuPDFLoader(path)
        docs = loader.load()
        combined = "\n\n".join([d.page_content for d in docs if d.page_content])
        if combined.strip():
            return combined
    except Exception:
        pass
    try:
        reader = PyPDF2.PdfReader(open(path, "rb"))
        return "\n\n".join([p.extract_text() or "" for p in reader.pages])
    except Exception:
        return ""

# ---------------------------
# Ingest
# ---------------------------
def ingest_documents(chroma, session_id: str, files: List[Dict]):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    texts, metas, ids = [], [], []
    db = DBSession()
    for f in files:
        doc_id = str(uuid.uuid4())
        text = pdf_to_text(f["path"])
        if not text.strip():
            continue
        chunks = splitter.split_text(text)
        for i, c in enumerate(chunks):
            texts.append(c)
            metas.append({"session_id": session_id, "doc_id": doc_id, "filename": f["filename"], "source": f["source"]})
            ids.append(f"{doc_id}_{i}")
        db.add(DocumentRecord(id=doc_id, session_id=session_id, filename=f["filename"], source=f["source"]))
    db.commit(); db.close()
    if texts:
        chroma.add_texts(texts=texts, metadatas=metas, ids=ids)
        chroma.persist()

# ---------------------------
# Session helpers
# ---------------------------
def create_session(name="Untitled Chat"):
    db = DBSession()
    sid = str(uuid.uuid4())
    db.add(SessionRecord(id=sid, name=name))
    db.commit(); db.close()
    return sid

def rename_session(sid, new):
    db = DBSession(); s = db.query(SessionRecord).get(sid)
    if s: s.name = new; db.commit()
    db.close()

def delete_session(sid):
    db = DBSession(); s = db.query(SessionRecord).get(sid)
    if s: db.delete(s); db.commit()
    db.close()

def save_message(sid, role, content):
    db = DBSession()
    db.add(MessageRecord(session_id=sid, role=role, content=content))
    sess = db.query(SessionRecord).get(sid)
    if sess: sess.last_active = datetime.utcnow()
    db.commit(); db.close()

def load_messages(sid):
    db = DBSession()
    msgs = db.query(MessageRecord).filter_by(session_id=sid).order_by(MessageRecord.created_at).all()
    db.close()
    return [{"role": m.role, "content": m.content} for m in msgs]

def rehydrate_memory(sid):
    mem = ConversationBufferMemory(memory_key="chat_history", input_key="question", return_messages=True)
    for m in load_messages(sid):
        if m["role"] in ["human", "user"]: mem.chat_memory.add_user_message(m["content"])
        elif m["role"] in ["ai", "assistant"]: mem.chat_memory.add_ai_message(m["content"])
    return mem

# ---------------------------
# RAG
# ---------------------------
def retrieve_docs(chroma, sid, q, k=5):
    docs = chroma.as_retriever(search_kwargs={"k": k}).get_relevant_documents(q)
    return [d for d in docs if d.metadata.get("session_id") == sid]

def ask_question(chroma, sid, mem, q):
    save_message(sid, "human", q); mem.chat_memory.add_user_message(q)
    ctx = "\n\n".join(d.page_content for d in retrieve_docs(chroma, sid, q))
    hist = "\n".join([f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}" for m in mem.chat_memory.messages[:-1]])
    ans = call_groq_llm(f"Context:\n{ctx}\n\nHistory:\n{hist}\n\nQ: {q}\nA:")
    save_message(sid, "ai", ans); mem.chat_memory.add_ai_message(ans)
    return ans

# ---------------------------
# UI
# ---------------------------
st.set_page_config(page_title="Research Paper Deconstructor", layout="wide")
st.title("Research Paper Deconstructor")

chroma = init_chroma()
if "sid" not in st.session_state: st.session_state.sid = None
if "mem" not in st.session_state: st.session_state.mem = None
if "menu_open" not in st.session_state: st.session_state.menu_open = None

# LEFT SIDEBAR
with st.sidebar:
    st.subheader("Configuration")
    api_key_input = st.text_input("Groq API Key", type="password")
    if api_key_input: 
        os.environ["GROQ_API_KEY"] = api_key_input
        st.success("✅ API key configured")

    st.subheader("Chats")
    if st.button("New Chat", use_container_width=True):
        st.session_state.sid = create_session()
        st.session_state.mem = rehydrate_memory(st.session_state.sid)
        st.session_state.menu_open = None
        st.rerun()

    db = DBSession(); sessions = db.query(SessionRecord).order_by(SessionRecord.last_active.desc()).all(); db.close()
    for s in sessions:
        cols = st.columns([0.2, 4])
        with cols[0]:
            if st.button("⋮", key=f"menu_{s.id}"):
                st.session_state.menu_open = s.id if st.session_state.menu_open != s.id else None
                st.rerun()
        with cols[1]:
            if st.button(s.name or "Untitled", key=f"sel_{s.id}", use_container_width=True):
                st.session_state.sid = s.id
                st.session_state.mem = rehydrate_memory(s.id)
                st.session_state.menu_open = None
                st.rerun()
        if st.session_state.menu_open == s.id:
            action = st.radio("Action", ["Rename", "Delete"], key=f"act_{s.id}")
            if action == "Rename":
                new_name = st.text_input("New name:", value=s.name or "", key=f"ren_{s.id}")
                if st.button("Save", key=f"save_{s.id}"):
                    rename_session(s.id, new_name)
                    st.session_state.menu_open = None
                    st.rerun()
            elif action == "Delete":
                if st.button("Confirm Delete", key=f"del_{s.id}"):
                    delete_session(s.id)
                    if st.session_state.sid == s.id: st.session_state.sid = None; st.session_state.mem = None
                    st.session_state.menu_open = None
                    st.rerun()

# MAIN + RIGHT
if not st.session_state.sid:
    st.info("Start a new chat or select one.")
else:
    left, right = st.columns([3,1])

    with left:
        for m in load_messages(st.session_state.sid):
            with st.chat_message("user" if m["role"]=="human" else "assistant"): st.markdown(m["content"])
        if q := st.chat_input("Ask about your documents"):
            with st.chat_message("user"): st.markdown(q)
            with st.chat_message("assistant"): st.markdown(ask_question(chroma, st.session_state.sid, st.session_state.mem, q))

    with right:
        st.subheader("Attached PDFs")
        db = DBSession(); docs = db.query(DocumentRecord).filter_by(session_id=st.session_state.sid).all(); db.close()
        for d in docs: st.text(d.filename)
        st.subheader("Browse & Process")
        files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True, key="up")
        if st.button("Process", key="proc"):
            if files:
                to_ingest = []
                for f in files:
                    path = os.path.join(tempfile.gettempdir(), f.name)
                    with open(path,"wb") as out: out.write(f.getvalue())
                    to_ingest.append({"path": path, "filename": f.name, "source": "upload"})
                ingest_documents(chroma, st.session_state.sid, to_ingest)
                st.success("✅ Files processed"); st.rerun()
