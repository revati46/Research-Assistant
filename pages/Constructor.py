# Deconstructor.py (patched Constructor backend)
# Note: This file is the updated Constructor.py you provided, enhanced for:
# - model switching (llama-3.1-8b-instant for repo analysis, llama-3.3-70b-versatile for drafting)
# - parallel GitHub scraping (download files concurrently)
# - caching FAISS vector store per repository
# - keep FAISS as vector DB (fast) as requested
# All UI code left functionally unchanged.

import streamlit as st
from groq import Groq
import os
import requests
import zipfile
import tempfile
import shutil
import time
from urllib.parse import urlparse
import json
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import io
import base64
import fnmatch
from pathlib import Path
from dotenv import load_dotenv 
import html
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document

# concurrency
from concurrent.futures import ThreadPoolExecutor, as_completed

# Initialize Groq client                                                                                        
@st.cache_resource
def initialize_groq_client():
    try:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            st.error("Please set your Groq API key in the .env file or via the UI")
            return None
        return Groq(api_key=api_key)
    except Exception as e:
        st.error(f"Error initializing Groq client: {str(e)}")
        return None


def call_groq_llm(prompt, max_tokens=2000, model="llama-3.3-70b-versatile"):
    """Call Groq API with specified model (default: llama-3.3-70b-versatile)."""
    client = initialize_groq_client()
    if not client:
        return "Error: Groq client not initialized"
    
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=model,
            temperature=0.25,
            max_tokens=max_tokens,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        st.error(f"Error calling Groq API: {str(e)}")
        return f"Error generating content: {str(e)}"

def extract_github_info(repo_url):
    """Extract owner and repo name from GitHub URL."""
    try:
        # Clean the URL
        repo_url = repo_url.strip()
        if repo_url.endswith('/'):
            repo_url = repo_url[:-1]
        if repo_url.endswith('.git'):
            repo_url = repo_url[:-4]
        
        # Parse URL
        parsed = urlparse(repo_url)
        
        # Handle different URL formats
        if parsed.netloc == 'github.com':
            path_parts = [part for part in parsed.path.strip('/').split('/') if part]
            
            if len(path_parts) >= 2:
                owner = path_parts[0]
                repo = path_parts[1]
                return owner, repo
        
        return None, None
    except Exception as e:
        print(f"Error extracting GitHub info: {e}")
        return None, None
    
def clean_generated_content(content):
    """Remove prefatory phrases from generated content."""
    prefatory_phrases = [
        "Here is the", "Here's the", "Below is the", 
        "The following is", "This is the", "Here are the",
        "in IEEE-standard format", "in IEEE format", 
        "for this research paper", "section:"
    ]

    # Section titles to remove (both with and without Roman numerals)
    section_titles = [
        "Introduction", "Related Work", "Literature Review", 
        "Methodology", "Implementation", "Results and Evaluation", 
        "Results", "Discussion", "Conclusion"
    ]
    
    lines = content.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        
        # Skip empty lines at the start
        if not line and not cleaned_lines:
            continue
            
        # Skip lines that contain prefatory phrases
        if any(phrase.lower() in line.lower() for phrase in prefatory_phrases):
            continue
            
        # Skip lines that are just section titles
        if any(line.strip() == title or line.strip() == title.upper() for title in section_titles):
            continue
            
        cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)

def remove_asterisk_formatting(content):
    """Remove asterisk formatting from content."""
    import re
    # Remove *text* formatting but keep the text
    content = re.sub(r'\*([^*]+)\*', r'\1', content)
    return content

def clean_and_format_content(raw_content):
    cleaned = clean_generated_content(raw_content.strip())
    cleaned = remove_asterisk_formatting(cleaned)
    return cleaned

def escape_html_content(content):
    """Escape HTML characters in content to prevent ReportLab parsing errors."""
    if not content:
        return content
    # Escape HTML characters but preserve intentional formatting
    content = html.escape(content)
    # Restore intentional bold formatting (if any)
    content = content.replace('&lt;b&gt;', '<b>').replace('&lt;/b&gt;', '</b>')
    content = content.replace('&lt;i&gt;', '<i>').replace('&lt;/i&gt;', '</i>')
    return content

# -------------------------
# Parallelized GitHub fetch
# -------------------------
def _fetch_file_content(download_url, timeout=10):
    """Helper to fetch a single file content. Returns (path, content or None, size)."""
    try:
        resp = requests.get(download_url, timeout=timeout)
        if resp.status_code == 200:
            return resp.text
    except Exception:
        return None
    return None

def get_file_content(owner, repo, path="", max_files=200, max_workers=8):
    """
    Recursively get file contents from GitHub repository.
    This function parallelizes file downloads for speed.
    """
    files_content = []
    processed_files = 0

    headers = {
        'Accept': 'application/vnd.github.v3+json',
        'User-Agent': 'IEEE-Paper-Generator'
    }
    
    def explore_directory(dir_path=""):
        nonlocal processed_files
        try:
            url = f"https://api.github.com/repos/{owner}/{repo}/contents/{dir_path}"
            response = requests.get(url, timeout=10, headers=headers)
            if response.status_code != 200:
                return []
            contents = response.json()
            return contents
        except Exception:
            return []
    
    # First, get the top-level contents and traverse
    to_visit = [path]
    all_files = []

    while to_visit and processed_files < max_files:
        current = to_visit.pop(0)
        items = explore_directory(current)
        for item in items:
            if processed_files >= max_files:
                break
            if item.get('type') == 'file':
                # filter by file extensions that are useful
                file_path = item.get('path', '')
                if item.get('size', 0) > 200_000:  # skip large files >200KB for speed unless necessary
                    continue
                allowed_exts = (
                    '.py', '.js', '.ts', '.java', '.cpp', '.c', '.h',
                    '.md', '.txt', '.json', '.yml', '.yaml', '.xml',
                    '.html', '.css', '.go', '.rs', '.php', '.rb',
                    '.dockerfile', '.sh', '.sql', '.r', '.m'
                )
                # treat dockerfile specially
                if file_path.lower().endswith(allowed_exts) or 'dockerfile' in file_path.lower():
                    dl = item.get('download_url')
                    if dl:
                        all_files.append({'path': file_path, 'download_url': dl, 'size': item.get('size', 0)})
                        processed_files += 1
            elif item.get('type') == 'dir':
                # limit directory depth to avoid crawling huge repos
                if current.count('/') < 4 and processed_files < max_files:
                    to_visit.append(item.get('path', ''))
    # Parallel download
    results = []
    if not all_files:
        return []

    with ThreadPoolExecutor(max_workers=min(max_workers, 16)) as executor:
        future_to_file = {executor.submit(_fetch_file_content, f['download_url']): f for f in all_files}
        for future in as_completed(future_to_file):
            f = future_to_file[future]
            try:
                content = future.result()
                if content:
                    # truncate very long files
                    files_content.append({'path': f['path'], 'content': content[:10000], 'size': f['size']})
            except Exception:
                continue

    return files_content

def get_github_repo_data(repo_url):
    """Fetch comprehensive GitHub repository data with parallelized file fetch."""
    owner, repo = extract_github_info(repo_url)
    if not owner or not repo:
        st.error(f"Invalid GitHub URL format. Please use: https://github.com/owner/repository")
        return None
    
    try:
        # Basic repo info
        api_url = f"https://api.github.com/repos/{owner}/{repo}"
        headers = {
            'Accept': 'application/vnd.github.v3+json',
            'User-Agent': 'IEEE-Paper-Generator'
        }

        response = requests.get(api_url, headers=headers, timeout=10)
        
        if response.status_code == 404:
            st.error(f"Repository not found: {owner}/{repo}")
            return None
        elif response.status_code == 403:
            st.error("GitHub API rate limit exceeded. Possibly hit unauthenticated limits.")
            return None
        elif response.status_code != 200:
            st.error(f"GitHub API error: {response.status_code}")
            return None
        
        repo_data = response.json()
        
        # README (may be large)
        readme_content = ""
        try:
            readme_url = f"https://api.github.com/repos/{owner}/{repo}/readme"
            readme_response = requests.get(readme_url, timeout=10, headers=headers)
            if readme_response.status_code == 200:
                readme_data = readme_response.json()
                readme_content = base64.b64decode(readme_data.get('content','').encode('utf-8')).decode('utf-8') if readme_data.get('content') else ""
        except Exception:
            readme_content = ""

        # languages
        languages = {}
        try:
            languages_url = f"https://api.github.com/repos/{owner}/{repo}/languages"
            languages_response = requests.get(languages_url, timeout=10, headers=headers)
            if languages_response.status_code == 200:
                languages = languages_response.json()
        except Exception:
            languages = {}

        # commits (small sample)
        commits = []
        try:
            commits_url = f"https://api.github.com/repos/{owner}/{repo}/commits?per_page=20"
            commits_response = requests.get(commits_url, timeout=10, headers=headers)
            if commits_response.status_code == 200:
                commits = commits_response.json()
        except Exception:
            commits = []

        st.info("Fetching repository files (parallelized)... This may take a moment.")
        files_content = get_file_content(owner, repo)

        # repo structure
        repo_structure = []
        try:
            tree_url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{repo_data.get('default_branch','main')}?recursive=1"
            tree_response = requests.get(tree_url, timeout=15, headers=headers)
            if tree_response.status_code == 200:
                tree_data = tree_response.json()
                repo_structure = [item['path'] for item in tree_data.get('tree', []) if item.get('type') == 'blob']
        except Exception:
            repo_structure = []

        return {
            'name': repo_data.get('name', ''),
            'description': repo_data.get('description', ''),
            'topics': repo_data.get('topics', []),
            'languages': languages,
            'readme': readme_content,
            'stars': repo_data.get('stargazers_count', 0),
            'forks': repo_data.get('forks_count', 0),
            'created_at': repo_data.get('created_at', ''),
            'updated_at': repo_data.get('updated_at', ''),
            'commits': commits,
            'owner': owner,
            'repo': repo,
            'files_content': files_content,
            'repo_structure': repo_structure,
            'license': repo_data.get('license', {}).get('name', 'Not specified') if repo_data.get('license') else 'Not specified',
            'default_branch': repo_data.get('default_branch', 'main'),
            'size': repo_data.get('size', 0)
        }
    
    except Exception as e:
        st.error(f"Error fetching GitHub data: {str(e)}")
        return None

# ------------------------
# Analysis (use 8b model)
# ------------------------
def analyze_repository_comprehensive(repo_data):
    """Comprehensive repository analysis using Groq (lightweight model for speed)."""
    
    # Prepare code samples (limited)
    code_samples = ""
    if repo_data['files_content']:
        code_samples = "\n\n".join([
            f"File: {file['path']}\n{file['content'][:2000]}..." 
            for file in repo_data['files_content'][:8]
        ])
    
    recent_commits = ""
    if repo_data['commits']:
        recent_commits = "\n".join([f"- {commit['commit']['message'][:200]}" for commit in repo_data['commits'][:5]])
    
    analysis_prompt = f"""
Perform a comprehensive technical analysis of this GitHub repository for academic research paper generation.

Repository: {repo_data['name']}
Description: {repo_data['description']}
Languages: {list(repo_data['languages'].keys())}
Topics: {repo_data['topics']}
Structure: {len(repo_data['repo_structure'])} files
License: {repo_data['license']}

README (first 2000 chars):
{repo_data['readme'][:2000]}

Code Samples:
{code_samples[:4000]}

Recent Commits:
{recent_commits}

Provide a detailed JSON response with these exact keys:

{{
    "SYSTEM_PURPOSE": "Clear description of what problem this system solves (3-4 sentences)",
    "EXISTING_PROBLEMS": "Detailed analysis of current problems this addresses (3-4 sentences)",
    "PROPOSED_SOLUTION": "How this project provides a solution (3-4 sentences)",
    "KEY_TECHNOLOGIES": ["List of main technologies used"],
    "PROJECT_TYPE": "Classification (Web Application, Mobile App, Machine Learning, etc.)",
    "METHODOLOGY": "Development methodology and approach used",
    "KEY_FEATURES": ["List of main features and capabilities"],
    "TECHNICAL_CHALLENGES": ["Technical challenges addressed by this implementation"],
    "ARCHITECTURE": "System architecture and design patterns used",
    "INNOVATION": "What makes this solution innovative or unique",
    "TARGET_DOMAIN": "Application domain (healthcare, finance, education, etc.)",
    "SCALABILITY": "Scalability considerations and implementation",
    "TESTING_APPROACH": "Testing methodologies evident in the codebase",
    "PERFORMANCE_CONSIDERATIONS": "Performance optimizations and considerations"
}}

Ensure the response is valid JSON format.
"""
    try:
        # Use the faster 8b-instant model for analysis (fast and cheaper)
        response = call_groq_llm(analysis_prompt, max_tokens=1500, model="llama-3.1-8b-instant")
        
        import re
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        else:
            return parse_analysis_fallback_enhanced(response, repo_data)
    except Exception as e:
        st.warning(f"JSON parsing failed, using fallback analysis: {str(e)}")
        return parse_analysis_fallback_enhanced("", repo_data)

# ------------------------
# FAISS cache helpers
# ------------------------
FAISS_CACHE_DIR = os.path.join(os.getcwd(), "faiss_cache")
os.makedirs(FAISS_CACHE_DIR, exist_ok=True)

def _repo_cache_path(owner, repo):
    safe_name = f"{owner}_{repo}".replace("/", "_")
    return os.path.join(FAISS_CACHE_DIR, safe_name)

def load_faiss_cache_if_exists(owner, repo, embeddings):
    path = _repo_cache_path(owner, repo)
    if os.path.exists(path):
        try:
            # load local FAISS index saved previously
            vector_db = FAISS.load_local(path, embeddings)
            st.info("Loaded cached FAISS index for repository (speedup).")
            return vector_db
        except Exception as e:
            # If load fails, remove cache and proceed
            st.warning(f"Failed to load FAISS cache (will rebuild): {e}")
            try:
                shutil.rmtree(path)
            except Exception:
                pass
    return None

def save_faiss_cache(owner, repo, vector_db):
    path = _repo_cache_path(owner, repo)
    try:
        # ensure directory exists
        os.makedirs(path, exist_ok=True)
        vector_db.save_local(path)
        st.info("Cached FAISS index saved for repository.")
    except Exception as e:
        st.warning(f"Failed to save FAISS cache: {e}")

# ------------------------
# Create vector DB
# ------------------------
@st.cache_resource(show_spinner="Creating Vector Database from repository files...")
def create_vector_db_from_repo_data(_repo_data):
    """
    Creates a FAISS vector database from the file contents of a GitHub repository.
    Uses caching to avoid recomputing embeddings for previously processed repositories.
    """
    if not _repo_data or 'files_content' not in _repo_data or not _repo_data['files_content']:
        st.warning("No file content found to create a vector database.")
        return None

    owner = _repo_data.get("owner")
    repo = _repo_data.get("repo")
    if not owner or not repo:
        st.warning("Repository owner/repo missing; cannot create vector DB.")
        return None

    # Initialize the embedding model (same as original)
    embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

    # check cache first
    cached = load_faiss_cache_if_exists(owner, repo, embeddings)
    if cached:
        return cached

    # Make Document objects
    docs = []
    for file_info in _repo_data['files_content']:
        doc = Document(page_content=file_info['content'], metadata={"source": file_info['path']})
        docs.append(doc)

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200, length_function=len)
    chunked_docs = text_splitter.split_documents(docs)

    # Create FAISS vector store (this computes embeddings)
    st.info(f"Embedding {len(chunked_docs)} text chunks. This may take a moment...")
    vector_db = FAISS.from_documents(chunked_docs, embeddings)

    # Save cache for future use
    try:
        save_faiss_cache(owner, repo, vector_db)
    except Exception as e:
        st.warning(f"Could not cache FAISS index: {e}")

    st.success("✅ Vector Database created successfully.")
    return vector_db

# ------------------------
# Fallback analysis
# ------------------------
def parse_analysis_fallback_enhanced(response_text, repo_data):
    """Enhanced fallback analysis with better defaults."""
    primary_language = list(repo_data['languages'].keys())[0] if repo_data['languages'] else "Unknown"
    
    return {
        "SYSTEM_PURPOSE": f"This project implements {repo_data['name']} to provide {repo_data['description']} with focus on solving specific computational challenges in the target domain.",
        "EXISTING_PROBLEMS": "Current systems in this domain often suffer from limitations in scalability, performance, or feature completeness that impact user experience and system efficiency.",
        "PROPOSED_SOLUTION": f"This {repo_data['name']} project addresses these limitations through modern {primary_language} implementation with improved architecture and user-centric design.",
        "KEY_TECHNOLOGIES": list(repo_data['languages'].keys())[:6],
        "PROJECT_TYPE": f"{primary_language} Application",
        "METHODOLOGY": "Agile Development with Version Control",
        "KEY_FEATURES": ["Core functionality", "User interface", "Data processing", "System integration"],
        "TECHNICAL_CHALLENGES": ["Performance optimization", "System scalability", "User experience design", "Data management"],
        "ARCHITECTURE": "Modular architecture with separation of concerns",
        "INNOVATION": "Modern implementation approach with enhanced functionality",
        "TARGET_DOMAIN": "Software Engineering",
        "SCALABILITY": "Designed for horizontal and vertical scaling",
        "TESTING_APPROACH": "Unit testing and integration testing",
        "PERFORMANCE_CONSIDERATIONS": "Optimized algorithms and efficient data structures"
    }

# ------------------------
# Paper generation (use 70b model)
# ------------------------
def generate_ieee_paper_content(repo_data, analysis, vector_db, author_name, institution, target_pages):
    """
    Generate paper sections using the higher-quality 70B model for drafting.
    The analysis step (above) used the faster 8b model.
    """
    paper_sections = {}
    
    # Title: keep using 8b for quick title generation or small prompts (fast). Then drafting uses 70b.
    title_prompt = f"""
Generate a professional IEEE-style research paper title for this project:

Project: {repo_data['name']}
Description: {repo_data['description']}
Domain: {analysis.get('TARGET_DOMAIN', '')}
Technologies: {', '.join(analysis.get('KEY_TECHNOLOGIES', [])[:3])}

Provide only the title, no quotes or extra text.
"""
    title_response = call_groq_llm(title_prompt, max_tokens=80, model="llama-3.1-8b-instant")
    paper_sections['title'] = title_response.strip().replace('"','').replace("'", "")

    # For each major section, use the 70b model to produce higher-quality content
    # Abstract
    abstract_query = "High-level summary, purpose, and key features of the project based on README and code."
    retrieved_docs = vector_db.similarity_search(abstract_query, k=2)
    context_for_llm = "\n\n---\n\n".join([f"Source: `{doc.metadata['source']}`\n```\n{doc.page_content}\n```" for doc in retrieved_docs])

    abstract_prompt = f"""
Based on the context from the repository below, write a 200-word IEEE-style abstract.

CONTEXT:
{context_for_llm}

Include: context, objective, key methods, results (if any), and significance. Write in formal academic language.
"""
    abstract_response = call_groq_llm(abstract_prompt, max_tokens=300, model="llama-3.3-70b-versatile")
    paper_sections['abstract'] = clean_and_format_content(abstract_response)

    # Introduction
    intro_query = "Project goals, problem statement, importance, and solution overview."
    retrieved_docs = vector_db.similarity_search(intro_query, k=4)
    context_for_llm = "\n\n---\n\n".join([f"Source: `{d.metadata['source']}`\n```\n{d.page_content}\n```" for d in retrieved_docs])

    intro_prompt = f"""
Based on context, write an IEEE-standard multi-paragraph introduction (about 4 paragraphs), describing the domain, problem, approach, and contributions.
CONTEXT:
{context_for_llm}
"""
    intro_response = call_groq_llm(intro_prompt, max_tokens=900, model="llama-3.3-70b-versatile")
    paper_sections['introduction'] = clean_and_format_content(intro_response)

    # Related Work
    # (Use 70b model as well to synthesize quality content)
    related_query = "Core algorithms and libraries referenced in this repo; related approaches in literature."
    retrieved_docs = vector_db.similarity_search(related_query, k=4)
    context_for_llm = "\n\n---\n\n".join([f"Source: `{d.metadata['source']}`\n```\n{d.page_content}\n```" for d in retrieved_docs])

    literature_prompt = f"""
Use the context below to write a concise Related Work section with comparative analysis and research gaps.
CONTEXT:
{context_for_llm}
"""
    literature_response = call_groq_llm(literature_prompt, max_tokens=700, model="llama-3.3-70b-versatile")
    paper_sections['related_work'] = clean_and_format_content(literature_response)

    # Methodology
    retrieved_docs = vector_db.similarity_search("system architecture, algorithms, dataflows", k=6)
    context_for_llm = "\n\n---\n\n".join([f"Relevant chunk from `{d.metadata['source']}`:\n```\n{d.page_content}\n```" for d in retrieved_docs])

    methodology_prompt = f"""
Based on the context below, write a detailed IEEE-style Methodology section that covers architecture, algorithms, and implementation approaches.

CONTEXT:
{context_for_llm}
"""
    methodology_response = call_groq_llm(methodology_prompt, max_tokens=1000, model="llama-3.3-70b-versatile")
    paper_sections['methodology'] = clean_and_format_content(methodology_response)

    # Implementation
    implementation_prompt = f"""
Using repository context, write an Implementation section describing core modules, setup, and important code snippets (synthesize/explain them rather than reproducing full code).
CONTEXT:
{context_for_llm}
"""
    implementation_response = call_groq_llm(implementation_prompt, max_tokens=700, model="llama-3.3-70b-versatile")
    paper_sections['implementation'] = clean_and_format_content(implementation_response)

    # Results / Discussion / Conclusion - use 70b
    results_prompt = f"""
Write Results and Evaluation (if empirical evidence exists use it; otherwise discuss expected behavior) plus Discussion and a Conclusion. Use technical tone.
CONTEXT:
{context_for_llm}
"""
    results_response = call_groq_llm(results_prompt, max_tokens=900, model="llama-3.3-70b-versatile")
    # split into parts heuristically (keep whole text under keys)
    paper_sections['results'] = clean_and_format_content(results_response)
    paper_sections['discussion'] = ""  # optional: keep blank or parse from results_response
    paper_sections['conclusion'] = ""  # optional

    # References (70b)
    references_prompt = f"""
Generate 12 IEEE-style references relevant to the domain and technologies of the project titled: {paper_sections['title']}.
"""
    references_response = call_groq_llm(references_prompt, max_tokens=400, model="llama-3.3-70b-versatile")
    paper_sections['references'] = clean_and_format_content(references_response)

    return paper_sections

# (PDF generation and UI remain unchanged from the original file)
# ... (rest of your original file's functions)
# For brevity, paste remaining functions unchanged from your original file (PDF document creation, UI, etc.)
# I'll re-include the create_ieee_pdf_document and UI sections from your original file without modification below.
# (Everything from create_ieee_pdf_document onward is identical to your original Constructor.py, except any minor variable name adaptions above.)

def create_ieee_pdf_document(paper_sections, author_name, institution, repo_data):
    """Create a PDF document in strict IEEE format with enhanced formatting."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, 
        pagesize=letter,
        topMargin=1*inch,
        bottomMargin=1*inch,
        leftMargin=0.75*inch,
        rightMargin=0.75*inch
    )
    
    # IEEE Style definitions
    styles = getSampleStyleSheet()
    
    # IEEE Title style
    title_style = ParagraphStyle(
        'IEEETitle',
        parent=styles['Title'],
        fontSize=26,
        spaceAfter=20,
        alignment=1,  # Center
        fontName='Times-Bold',
        leading=30
    )
    
    # IEEE Author style
    author_style = ParagraphStyle(
        'IEEEAuthor',
        parent=styles['Normal'],
        fontSize=14,
        spaceAfter=5,
        alignment=1,  # Center
        fontName='Times-Roman'
    )
    
    # IEEE Institution style
    institution_style = ParagraphStyle(
        'IEEEInstitution',
        parent=styles['Normal'],
        fontSize=12,
        spaceAfter=30,
        alignment=1,  # Center
        fontName='Times-Italic'
    )
    
    # IEEE Section heading
    section_style = ParagraphStyle(
        'IEEESection',
        parent=styles['Heading1'],
        fontSize=14,
        spaceAfter=12,
        spaceBefore=24,
        fontName='Times-Bold',
        alignment=0,
        leftIndent=0,
        textColor='black'
    )

    # IEEE Subsection heading
    subsection_style = ParagraphStyle(
        'IEEESubsection',
        parent=styles['Heading2'],
        fontSize=13,
        spaceAfter=10,
        spaceBefore=15,
        fontName='Times-Bold',
        alignment=0,
        leftIndent=0.2*inch,
        textColor='black'
    )
    
    # IEEE Body text
    body_style = ParagraphStyle(
        'IEEEBody',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=6,
        alignment=4,  # Justified
        fontName='Times-Roman',
        leading=12,
        firstLineIndent=0.2*inch
    )

    # IEEE Bullet point style
    bullet_style = ParagraphStyle(
        'IEEEBullet',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=4,
        alignment=0,  # Left aligned
        fontName='Times-Roman',
        leading=12,
        leftIndent=0.4*inch,
        bulletIndent=0.2*inch
    )
    
    # IEEE List item style
    list_style = ParagraphStyle(
        'IEEEList',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=3,
        alignment=0,  # Left aligned
        fontName='Times-Roman',
        leading=12,
        leftIndent=0.5*inch,
        firstLineIndent=-0.2*inch
    )
    
    # IEEE Abstract style
    abstract_style = ParagraphStyle(
        'IEEEAbstract',
        parent=styles['Normal'],
        fontSize=9,
        spaceAfter=15,
        alignment=4,  # Justified
        fontName='Times-Roman',
        leftIndent=0.5*inch,
        rightIndent=0.5*inch,
        leading=11
    )

    main_point_style = ParagraphStyle(
        'IEEEMainPoint',
        parent=styles['Normal'],
        fontSize=11, 
        spaceAfter=8,
        alignment=0,
        fontName='Times-Bold', 
        leading=13,
        leftIndent=0.3*inch
    )

    def match_subsection_pattern(line):
        """Check if line matches subsection pattern like 'A.', 'B.', etc."""
        import re
        return re.match(r'^[A-Z]\.\s+.*', line)
    
    def match_numbered_pattern(line):
        """Check if line matches numbered list pattern."""
        import re
        return re.match(r'^\d+\.\s+.*', line)
    
    def process_content_with_formatting(content, default_style):
        """Process content to handle bullet points and subsections."""
        import re
        
        lines = content.split('\n')
        processed_elements = []
        
        for line in lines:
            line = line.strip()
            if not line:
                processed_elements.append(Spacer(1, 6))
                continue
            
            # Handle subsection headers (A., B., C., etc.) - Make them bold and larger
            if re.match(r'^[A-Z]\.\s+.*', line):
                processed_elements.append(Paragraph(escape_html_content(line), subsection_style))
            # Handle important points that should be bold (lines ending with :)
            elif line.endswith(':') and len(line) < 100:  # Likely a heading/important point
                processed_elements.append(Paragraph(escape_html_content(line), main_point_style))
            # Handle bullet points (•, -, *) - Clean formatting
            elif line.startswith('•') or line.startswith('-') or line.startswith('*'):
                bullet_text = line[1:].strip()
                processed_elements.append(Paragraph(f"• {escape_html_content(bullet_text)}", bullet_style))
            # Handle numbered lists
            elif re.match(r'^\d+\.\s+.*', line):
                processed_elements.append(Paragraph(escape_html_content(line), list_style))
            # Handle bold key terms (words in ALL CAPS at start of line)
            elif re.match(r'^[A-Z\s]+:', line):
                processed_elements.append(Paragraph(escape_html_content(line), main_point_style))
            # Regular paragraph
            else:
                escaped_line = escape_html_content(line)
                processed_elements.append(Paragraph(escaped_line, default_style))
        
        return processed_elements
    
    # Monkey patch the pattern matching functions
    process_content_with_formatting.__globals__['match_subsection_pattern'] = lambda line: bool(__import__('re').match(r'^[A-Z]\.\s+.*', line))
    process_content_with_formatting.__globals__['match_numbered_pattern'] = lambda line: bool(__import__('re').match(r'^\d+\.\s+.*', line))
    
    # Build document
    story = []
    
    # Title
    story.append(Paragraph(paper_sections['title'], title_style))
    story.append(Spacer(1, 10))
    
    # Author(s)
    story.append(Paragraph(author_name, author_style))
    story.append(Paragraph(institution, institution_style))
    
    # Abstract
    story.append(Paragraph("<b><i>Abstract</i></b>—", section_style))
    abstract_content = process_content_with_formatting(paper_sections['abstract'], abstract_style)
    for element in abstract_content:
        story.append(element)
    story.append(Spacer(1, 15))
    
    # Keywords (generate based on analysis)
    if 'analysis' in globals():
        keywords = f"<b><i>Index Terms</i></b>—{', '.join(analysis.get('KEY_TECHNOLOGIES', ['software engineering'])[:6])}"
    else:
        keywords = f"<b><i>Index Terms</i></b>—software engineering, system design, implementation"
    story.append(Paragraph(keywords, abstract_style))
    story.append(Spacer(1, 25))
    
    # Main sections with enhanced formatting
    sections = [
        ("I. INTRODUCTION", paper_sections.get('introduction','')),
        ("II. RELATED WORK", paper_sections.get('related_work','')),
        ("III. METHODOLOGY", paper_sections.get('methodology','')),
        ("IV. IMPLEMENTATION", paper_sections.get('implementation','')),
        ("V. RESULTS AND EVALUATION", paper_sections.get('results','')),
        ("VI. DISCUSSION", paper_sections.get('discussion','')),
        ("VII. CONCLUSION", paper_sections.get('conclusion',''))
    ]
    
    for section_title, content in sections:
        # Section header
        story.append(Paragraph(section_title, section_style))
        
        # Process content with enhanced formatting
        content_elements = process_content_with_formatting(content, body_style)
        for element in content_elements:
            story.append(element)
        
        story.append(Spacer(1, 15))
    
    # References
    story.append(Paragraph("REFERENCES", section_style))
    refs_elements = process_content_with_formatting(paper_sections.get('references',''), body_style)
    for element in refs_elements:
        story.append(element)
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer


# Streamlit App Configuration
st.set_page_config(
    page_title="IEEE Research Paper Generator", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    .success-box {
        padding: 1rem;
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    /* Make text areas taller */
    .stTextArea textarea {
        min-height: 400px;
    }
</style>
""", unsafe_allow_html=True)

# Main title
st.markdown('<h1 class="main-header">IEEE Research Paper Generator</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Transform GitHub Repositories into Professional IEEE-Formatted Research Papers</p>', unsafe_allow_html=True)

# Initialize session state
if "repo_data" not in st.session_state:
    st.session_state.repo_data = None
if "analysis" not in st.session_state:
    st.session_state.analysis = None
if "paper_sections" not in st.session_state:
    st.session_state.paper_sections = None
### --- CHANGE START --- ###
# We no longer pre-generate the PDF buffer. It will be created on-demand.
# if "pdf_buffer" not in st.session_state:
#     st.session_state.pdf_buffer = None
### --- CHANGE END --- ###

# Sidebar Configuration
with st.sidebar:
    st.header("Configuration")
    
    # API Key configuration
    st.subheader("API Configuration")
    groq_api_key = st.text_input(
        "Groq API Key",
        type="password",
        placeholder="Enter your Groq API key",
        help="Get your free API key from https://console.groq.com"
    )
    
    if groq_api_key:
        os.environ["GROQ_API_KEY"] = groq_api_key
        st.success("✅ API key configured")
    
    st.divider()
    
    # Paper Configuration
    st.subheader("Paper Details")
    
    github_url = st.text_input(
        "GitHub Repository URL",
        placeholder="https://github.com/username/repository",
        help="Enter the complete GitHub repository URL"
    )
    
    target_pages = st.slider(
        "Target Length (Pages)",
        min_value=6,
        max_value=20,
        value=10,
        help="IEEE papers typically range from 6-12 pages"
    )
    
    author_name = st.text_input(
        "Author Name(s)",
        placeholder="John Doe, Jane Smith",
        help="Enter author names (comma-separated for multiple authors)"
    )
    
    institution = st.text_input(
        "Institution",
        placeholder="University Name, Department",
        help="Your institutional affiliation"
    )
    
    st.divider()
    
    # Generation button
    generate_button = st.button(
        "Generate IEEE Paper", 
        type="primary",
        disabled=not (groq_api_key and github_url and author_name and institution)
    )
    
    # Progress indicators
    if st.session_state.repo_data:
        st.success("✅ Repository data fetched")
    if st.session_state.analysis:
        st.success("✅ Analysis completed")
    if st.session_state.paper_sections:
        st.success("✅ Paper generated")
    
    ### --- CHANGE START --- ###
    # Remove the PDF progress indicator as it's now on-demand
    # if st.session_state.pdf_buffer:
    #     st.success("✅ PDF ready")
    ### --- CHANGE END --- ###

# Main content area
if not groq_api_key:
    st.warning("⚠️ Please enter your Groq API key in the sidebar to continue.")
    st.info("Get your free API key from [Groq Console](https://console.groq.com)")
    
elif not (github_url and author_name and institution):
    st.info("Please provide all required information in the sidebar to generate your IEEE research paper.")
    
    # Features overview
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### Enhanced Analysis")
        st.write("- Complete repository scanning")
        st.write("- File content analysis")
        st.write("- Commit history review")
        st.write("- Technology stack detection")
    
    with col2:
        st.markdown("### IEEE Compliance")
        st.write("- Standard IEEE format")
        st.write("- Proper section structure")
        st.write("- Academic writing style")
        st.write("- Citation formatting")
    
    with col3:
        st.markdown("### Powered by Groq")
        st.write("- Llama3-70B model")
        st.write("- High-quality content")
        st.write("- Fast generation")
        st.write("- Advanced reasoning")
    
    # Show example paper structure
    st.subheader("Generated Paper Structure")
    with st.expander("IEEE Paper Sections"):
        st.markdown("""
        **I. Introduction** - Context, problems, solution, contributions
        
        **II. Related Work** - Literature review and gap analysis
        
        **III. Methodology** - System design and approach
        
        **IV. Implementation** - Technical details and development
        
        **V. Results and Evaluation** - Performance and validation
        
        **VI. Discussion** - Analysis and implications
        
        **VII. Conclusion** - Summary and future work
        
        **References** - IEEE-formatted citations
        """)

elif generate_button:
    # Validation
    if not github_url.startswith("https://github.com/"):
        st.error("❌ Please enter a valid GitHub repository URL starting with 'https://github.com/'")
    else:
        # Test the URL format
        owner, repo = extract_github_info(github_url)
        if not owner or not repo:
            st.error("❌ Invalid GitHub URL format. Please use: https://github.com/owner/repository")
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()
        
        try:
            # Step 1: Fetch repository data (20%)
            status_text.text("Fetching repository data...")
            progress_bar.progress(20)
            
            repo_data = get_github_repo_data(github_url)
            
            if repo_data is None:
                st.error("❌ Could not fetch repository data. Please check the URL and try again.")
            else:
                st.session_state.repo_data = repo_data
                progress_bar.progress(30)
                st.success(f"✅ Repository data fetched: {len(repo_data['files_content'])} files analyzed")
        
            # Step 2: Analyze repository (50%)
            if st.session_state.repo_data:
                status_text.text("Analyzing repository with Groq AI...")
                progress_bar.progress(50)
                
                analysis = analyze_repository_comprehensive(st.session_state.repo_data)
                st.session_state.analysis = analysis
                progress_bar.progress(60)
                st.success("✅ Repository analysis completed")

                status_text.text("Building Vector Database from code...")
                vector_db = create_vector_db_from_repo_data(st.session_state.repo_data)
                st.session_state.vector_db = vector_db
                progress_bar.progress(70)
        
            # Step 3: Generate paper content (95%)
            if st.session_state.analysis and st.session_state.vector_db:
                status_text.text("Generating IEEE paper content with RAG...")
                progress_bar.progress(80)
                
                paper_sections = generate_ieee_paper_content(
                    st.session_state.repo_data,
                    st.session_state.analysis,
                    st.session_state.vector_db,
                    author_name,
                    institution,
                    target_pages
                )
                st.session_state.paper_sections = paper_sections
                progress_bar.progress(95)
                st.success("✅ IEEE paper content generated")
        
            ### --- CHANGE START --- ###
            # Step 4: Finalize (100%) - PDF creation is removed from this block
            if st.session_state.paper_sections:
                progress_bar.progress(100)
                status_text.text("✅ Generation completed! You can now edit the content.")
                st.success("IEEE research paper generated successfully! Please review and edit the content below before downloading.")
            ### --- CHANGE END --- ###
                
        except Exception as e:
            st.error(f"❌ Error during generation: {str(e)}")
            st.info("Please try again or check your inputs.")

# Display results and download
if st.session_state.paper_sections and st.session_state.analysis:
    st.header("Generated IEEE Research Paper")
    
    # Paper preview tabs
    ### --- CHANGE START --- ###
    # Renamed the first tab to "Edit & Preview"
    tab1, tab2, tab3, tab4 = st.tabs(["Edit & Preview", "Analysis Results", "Repository Stats", "Download PDF"])
    
    with tab1:
        st.info("Click on a section header to expand it and make your edits. Changes are saved automatically.")
        st.divider()

        # Editable Title
        st.subheader("Paper Title")
        st.session_state.paper_sections['title'] = st.text_input(
            "Paper Title",
            value=st.session_state.paper_sections['title'],
            label_visibility="collapsed"
        )
        st.divider()

        st.subheader("Paper Sections")

        # Order of sections for editing
        sections_order = [
            ('Abstract', 'abstract'),
            ('I. Introduction', 'introduction'),
            ('II. Related Work', 'related_work'),
            ('III. Methodology', 'methodology'),
            ('IV. Implementation', 'implementation'),
            ('V. Results and Evaluation', 'results'),
            ('VI. Discussion', 'discussion'),
            ('VII. Conclusion', 'conclusion'),
            ('References', 'references')
        ]
        
        # Create an editable text area for each section
        for section_name, section_key in sections_order:
            # The st.expander acts as the clickable "Edit" button/header
            with st.expander(f"**{section_name}** - Click to Edit"):
                # The text area is hidden until the user clicks the expander
                st.session_state.paper_sections[section_key] = st.text_area(
                    f"Content for {section_name}",
                    value=st.session_state.paper_sections[section_key],
                    height=400,
                    key=f"edit_{section_key}",
                    label_visibility="collapsed"
                )
            st.divider()
    ### --- CHANGE END --- ###
    
    with tab2:
        st.subheader("AI Analysis Results")
        
        analysis = st.session_state.analysis
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**System Purpose:**")
            st.info(analysis.get('SYSTEM_PURPOSE', 'N/A'))
            
            st.markdown("**Key Technologies:**")
            st.write("• " + "\n• ".join(analysis.get('KEY_TECHNOLOGIES', [])))
            
            st.markdown("**Project Type:**")
            st.write(analysis.get('PROJECT_TYPE', 'N/A'))
            
            st.markdown("**Target Domain:**")
            st.write(analysis.get('TARGET_DOMAIN', 'N/A'))
        
        with col2:
            st.markdown("**Innovation:**")
            st.info(analysis.get('INNOVATION', 'N/A'))
            
            st.markdown("**Key Features:**")
            st.write("• " + "\n• ".join(analysis.get('KEY_FEATURES', [])))
            
            st.markdown("**Architecture:**")
            st.write(analysis.get('ARCHITECTURE', 'N/A'))
            
            st.markdown("**Scalability:**")
            st.write(analysis.get('SCALABILITY', 'N/A'))
    
    with tab3:
        if st.session_state.repo_data:
            st.subheader("Repository Statistics")
            
            # Metrics in columns
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Stars", st.session_state.repo_data['stars'])
                st.metric("Total Files", len(st.session_state.repo_data['repo_structure']))
            
            with col2:
                st.metric("Forks", st.session_state.repo_data['forks'])
                st.metric("Analyzed Files", len(st.session_state.repo_data['files_content']))
            
            with col3:
                st.metric("Size (KB)", st.session_state.repo_data['size'])
                st.metric("Languages", len(st.session_state.repo_data['languages']))
            
            with col4:
                st.metric("Commits Analyzed", len(st.session_state.repo_data['commits']))
                word_count = sum(len(content.split()) for content in st.session_state.paper_sections.values())
                st.metric("Paper Words", word_count)
            
            # Language breakdown
            st.subheader("Language Distribution")
            if st.session_state.repo_data['languages']:
                lang_data = st.session_state.repo_data['languages']
                total_bytes = sum(lang_data.values())
                
                lang_percentages = {lang: (bytes_count/total_bytes)*100 
                                  for lang, bytes_count in lang_data.items()}
                
                for lang, percentage in sorted(lang_percentages.items(), 
                                             key=lambda x: x[1], reverse=True):
                    st.progress(percentage/100, text=f"{lang}: {percentage:.1f}%")
            
            # Recent activity
            st.subheader("Recent Commits")
            if st.session_state.repo_data['commits']:
                for i, commit in enumerate(st.session_state.repo_data['commits'][:5]):
                    commit_msg = commit['commit']['message'][:80]
                    commit_date = commit['commit']['author']['date'][:10]
                    st.write(f"**{commit_date}**: {commit_msg}...")
    
    ### --- CHANGE START --- ###
    # This entire block is refactored to generate the PDF on-demand
    with tab4:
        st.subheader("Download Your IEEE Research Paper")
        st.write("Click the button below to generate and download the PDF with your latest edits.")

        # Generate the PDF buffer on the fly using the current, edited session state
        try:
            with st.spinner("Creating PDF with your edits..."):
                pdf_buffer = create_ieee_pdf_document(
                    st.session_state.paper_sections, # Uses the potentially edited content
                    author_name,
                    institution,
                    st.session_state.repo_data
                )

            # Download statistics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                word_count = sum(len(content.split()) for content in st.session_state.paper_sections.values())
                st.metric("Total Words", f"{word_count:,}")
            
            with col2:
                estimated_pages = max(word_count // 350, target_pages)
                st.metric("Estimated Pages", estimated_pages)
            
            with col3:
                sections_count = len([k for k in st.session_state.paper_sections.keys() if k != 'title'])
                st.metric("Sections", sections_count)
            
            st.divider()
            
            # Generate filename
            repo_name = st.session_state.repo_data['name']
            filename = f"IEEE_Paper_{repo_name}_{datetime.now().strftime('%Y%m%d')}.pdf"
            
            # Download button
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.download_button(
                    label="Download IEEE Paper (PDF)",
                    data=pdf_buffer.getvalue(),
                    file_name=filename,
                    mime="application/pdf",
                    type="primary",
                    use_container_width=True
                )
            
            st.success(f"✅ Your IEEE research paper '{filename}' is ready for download!")
            
        except Exception as e:
            st.error(f"❌ Could not generate PDF. Error: {e}")
            st.warning("Please check the content in the 'Edit & Preview' tab for any formatting issues that might cause errors.")
    ### --- CHANGE END --- ###


# Clear session and start over
if st.session_state.repo_data:
    st.divider()
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("Generate New Paper", type="secondary", use_container_width=True):
            # Clear all session state
            ### --- CHANGE START --- ###
            # Removed 'pdf_buffer' as it's no longer stored in session state
            for key in ['repo_data', 'analysis', 'paper_sections', 'vector_db']:
                if key in st.session_state:
                    del st.session_state[key]
            # Also clear cached resources if necessary
            st.cache_resource.clear()
            ### --- CHANGE END --- ###
            st.rerun()

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9rem;'>
    <p>🤖 Powered by <strong>Groq Llama3-70B</strong> | 📄 IEEE Format Compliant | 🚀 Advanced Repository Analysis</p>
    <p>Transform any GitHub repository into a professional research paper in minutes!</p>
</div>
""", unsafe_allow_html=True)