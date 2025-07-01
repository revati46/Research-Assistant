# Home.py

import streamlit as st
from dotenv import load_dotenv

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Research Paper - Constructor & Deconstructor",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load environment variables from .env file
load_dotenv()

# --- HIDE SIDEBAR ON HOME PAGE ---
st.markdown("""
<style>
    [data-testid="stSidebar"], [data-testid="main-menu-button"] {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

# --- STYLING ---
st.markdown("""
<style>
    /* --- General & Theme --- */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .main > div {
        padding: 2rem 1rem 1rem 1rem;
    }
    .main-header {
        font-size: 3.5rem;
        font-weight: bold;
        color: #FFFFFF;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #A0A0A0;
        text-align: center;
        margin-bottom: 5rem;
    }

    /* --- Card Layout & Aspect Ratio --- */
    .card-grid-container {
        display: flex;          /* Use flexbox for robust column layout */
        justify-content: center;
        gap: 1.5rem;            /* ~1.5cm space between cards */
        max-width: 700px;       /* <<< KEY: This now reliably controls the overall width */
        margin: 0 auto;         /* Center the whole container */
    }
    
    .card-column {
        flex: 1; /* Each column will take up equal space */
    }
    
    a.card-link {
        text-decoration: none;
    }

    .card {
        position: relative;
        width: 100%;
        height: 0;
        padding-bottom: 75%; /* Wide 4:3 ratio */
        background-color: #161B22;
        border: 1px solid #30363D;
        border-radius: 16px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease, border-color 0.3s ease;
        overflow: hidden;
    }
    a.card-link:hover .card {
        transform: translateY(-8px);
        box-shadow: 0 12px 32px rgba(88, 166, 255, 0.2);
        border-color: #58A6FF;
    }
    .card-content {
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        padding: 1.5rem;
        text-align: center;
    }
    .card-title {
        font-size: 1.8rem;
        font-weight: bold;
        color: #58A6FF;
        margin-bottom: 0.75rem;
    }
    .card-description {
        font-size: 1rem;
        color: #C9D1D9;
        line-height: 1.4;
    }

    /* --- Footer --- */
    .footer {
        text-align: center;
        margin-top: 6rem;
        padding-top: 2rem;
        border-top: 1px solid #30363D;
        color: #8B949E;
        font-size: 1rem;
    }
    .tech-stack {
        font-weight: bold;
        color: #C9D1D9;
    }
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown('<h1 class="main-header">Research paper - Constructor & Deconstructor</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Build. Break. Understand.<br>AI-Powered Research Assistant.</p>', unsafe_allow_html=True)


# --- CARD LAYOUT ---
st.markdown("""
<div class="card-grid-container">
    <div class="card-column">
        <a href="/Constructor" class="card-link" target="_self">
            <div class="card">
                <div class="card-content">
                    <div class="card-title">Constructor</div>
                    <div class="card-description">
                        Transform your GitHub repositories into professionally formatted, IEEE-standard research papers.
                    </div>
                </div>
            </div>
        </a>
    </div>
    <div class="card-column">
        <a href="/Deconstructor" class="card-link" target="_self">
            <div class="card">
                <div class="card-content">
                    <div class="card-title">Deconstructor</div>
                    <div class="card-description">
                        Upload any research paper and start a conversation to extract key information instantly.
                    </div>
                </div>
            </div>
        </a>
    </div>
</div>
""", unsafe_allow_html=True)


# --- FOOTER ---
st.markdown(f"""
<div class="footer">
    <p>Powered by a modern AI stack including <span class="tech-stack">Streamlit, Groq, LangChain, Ollama, and Hugging Face.</span></p>
    <p>A comprehensive RAG project for academic and software engineering workflows.</p>
</div>
""", unsafe_allow_html=True)