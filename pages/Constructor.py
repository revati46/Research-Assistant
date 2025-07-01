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

# Initialize Groq client                                                                                        
@st.cache_resource
def initialize_groq_client():
    try:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            st.error("Please set your Groq API key in the .env file ")
            return None
        return Groq(api_key=api_key)
    except Exception as e:
        st.error(f"Error initializing Groq client: {str(e)}")
        return None


def call_groq_llm(prompt, max_tokens=2000):
    """Call Groq API with Llama3 model."""
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
            model="llama3-70b-8192",  # Using Llama3 70B model
            temperature=0.3,
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
    # Restore intentional bold formatting
    content = content.replace('<b>', '<b>').replace('</b>', '</b>')
    content = content.replace('<i>', '<i>').replace('</i>', '</i>')
    return content

def get_file_content(owner, repo, path="", max_files=50):
    """Recursively get file contents from GitHub repository."""
    files_content = []
    processed_files = 0

    # Add headers
    headers = {
        'Accept': 'application/vnd.github.v3+json',
        'User-Agent': 'IEEE-Paper-Generator'
    }
    
    def process_directory(dir_path=""):
        nonlocal processed_files
        if processed_files >= max_files:
            return
            
        try:
            url = f"https://api.github.com/repos/{owner}/{repo}/contents/{dir_path}"
            response = requests.get(url, timeout=10)
            
            if response.status_code != 200:
                return
            
            contents = response.json()
            
            for item in contents:
                if processed_files >= max_files:
                    break
                    
                if item['type'] == 'file':
                    # Skip binary files and large files
                    if item['size'] > 100000:  # Skip files larger than 100KB
                        continue
                        
                    file_path = item['path']
                    # Include important file types
                    if any(file_path.endswith(ext) for ext in [
                        '.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', 
                        '.md', '.txt', '.json', '.yml', '.yaml', '.xml',
                        '.html', '.css', '.go', '.rs', '.php', '.rb',
                        '.dockerfile', '.sh', '.sql', '.r', '.m'
                    ]):
                        try:
                            file_response = requests.get(item['download_url'], timeout=10)
                            if file_response.status_code == 200:
                                content = file_response.text[:5000]  # Limit content length
                                files_content.append({
                                    'path': file_path,
                                    'content': content,
                                    'size': item['size']
                                })
                                processed_files += 1
                        except:
                            continue
                            
                elif item['type'] == 'dir' and processed_files < max_files:
                    # Process subdirectories (limit depth)
                    if dir_path.count('/') < 3:  # Limit directory depth
                        process_directory(item['path'])
        except:
            pass
    
    process_directory(path)
    return files_content

def get_github_repo_data(repo_url):
    """Fetch comprehensive GitHub repository data."""
    owner, repo = extract_github_info(repo_url)
    if not owner or not repo:
        st.error(f"❌ Invalid GitHub URL format. Please use: https://github.com/owner/repository")
        return None
    
    try:
        # Basic repo info
        api_url = f"https://api.github.com/repos/{owner}/{repo}"

        # Add headers to avoid rate limiting
        headers = {
            'Accept': 'application/vnd.github.v3+json',
            'User-Agent': 'IEEE-Paper-Generator'
        }

        response = requests.get(api_url, headers=headers, timeout=10)
        
        if response.status_code == 404:
            st.error(f"❌ Repository not found: {owner}/{repo}")
            return None
        elif response.status_code == 403:
            st.error("❌ GitHub API rate limit exceeded. Please try again later.")
            return None
        elif response.status_code != 200:
            st.error(f"❌ GitHub API error: {response.status_code}")
            return None
        
        repo_data = response.json()
        
        # Get README content
        readme_url = f"https://api.github.com/repos/{owner}/{repo}/readme"
        readme_response = requests.get(readme_url, timeout=10)
        readme_content = ""
        
        if readme_response.status_code == 200:
            readme_data = readme_response.json()
            readme_content = base64.b64decode(readme_data['content']).decode('utf-8')
        
        # Get languages
        languages_url = f"https://api.github.com/repos/{owner}/{repo}/languages"
        languages_response = requests.get(languages_url, timeout=10)
        languages = {}
        
        if languages_response.status_code == 200:
            languages = languages_response.json()
        
        # Get recent commits
        commits_url = f"https://api.github.com/repos/{owner}/{repo}/commits?per_page=20"
        commits_response = requests.get(commits_url, timeout=10)
        commits = []
        
        if commits_response.status_code == 200:
            commits = commits_response.json()
        
        # Get repository structure and file contents
        st.info("Fetching repository files... This may take a moment.")
        files_content = get_file_content(owner, repo)
        
        # Get repository tree structure
        tree_url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{repo_data.get('default_branch', 'main')}?recursive=1"
        tree_response = requests.get(tree_url, timeout=15)
        repo_structure = []
        
        if tree_response.status_code == 200:
            tree_data = tree_response.json()
            repo_structure = [item['path'] for item in tree_data.get('tree', []) if item['type'] == 'blob']
        
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

def analyze_repository_comprehensive(repo_data):
    """Comprehensive repository analysis using Groq."""
    
    # Prepare code samples
    code_samples = ""
    if repo_data['files_content']:
        code_samples = "\n\n".join([
            f"File: {file['path']}\n{file['content'][:1000]}..." 
            for file in repo_data['files_content'][:10]
        ])
    
    # Prepare commit analysis
    recent_commits = ""
    if repo_data['commits']:
        recent_commits = "\n".join([
            f"- {commit['commit']['message'][:100]}" 
            for commit in repo_data['commits'][:5]
        ])
    
    analysis_prompt = f"""
    Perform a comprehensive technical analysis of this GitHub repository for academic research paper generation:

    Repository: {repo_data['name']}
    Description: {repo_data['description']}
    Languages: {list(repo_data['languages'].keys())}
    Topics: {repo_data['topics']}
    Structure: {len(repo_data['repo_structure'])} files
    License: {repo_data['license']}
    
    README Content (first 2000 chars):
    {repo_data['readme'][:2000]}
    
    Code Samples:
    {code_samples[:3000]}
    
    Recent Commits:
    {recent_commits}
    
    Based on this comprehensive analysis, provide a detailed JSON response with these exact keys:
    
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
        response = call_groq_llm(analysis_prompt, max_tokens=2000)
        
        # Try to extract JSON from response
        import re
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        else:
            # Fallback parsing
            return parse_analysis_fallback_enhanced(response, repo_data)
    except Exception as e:
        st.warning(f"JSON parsing failed, using fallback analysis: {str(e)}")
        return parse_analysis_fallback_enhanced("", repo_data)

@st.cache_resource(show_spinner="Creating Vector Database from repository files...")
def create_vector_db_from_repo_data(_repo_data):
    """
    Creates a FAISS vector database from the file contents of a GitHub repository.
    
    Args:
        _repo_data (dict): The dictionary containing repository data, including 'files_content'.

    Returns:
        FAISS: A FAISS vector store object.
    """
    if not _repo_data or 'files_content' not in _repo_data or not _repo_data['files_content']:
        st.warning("No file content found to create a vector database.")
        return None

    # Create LangChain Document objects from file contents
    docs = []
    for file_info in _repo_data['files_content']:
        # We create a Document for each file, adding the file path as metadata
        doc = Document(
            page_content=file_info['content'],
            metadata={"source": file_info['path']}
        )
        docs.append(doc)

    # Split the documents into smaller chunks for better retrieval
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        length_function=len
    )
    chunked_docs = text_splitter.split_documents(docs)

    # Initialize the embedding model. This runs locally on your machine.
    # 'all-MiniLM-L6-v2' is a good balance of speed and quality.
    embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
    
    # Create the FAISS vector store from the chunked documents and embeddings
    # This process can take a moment as it's calculating embeddings for all text chunks.
    st.info(f"Embedding {len(chunked_docs)} text chunks. This may take a moment...")
    vector_db = FAISS.from_documents(chunked_docs, embeddings)
    st.success("✅ Vector Database created successfully.")
    
    return vector_db

def parse_analysis_fallback_enhanced(response_text, repo_data):
    """Enhanced fallback analysis with better defaults."""
    primary_language = list(repo_data['languages'].keys())[0] if repo_data['languages'] else "Unknown"
    
    return {
        "SYSTEM_PURPOSE": f"This project implements {repo_data['name']} to provide {repo_data['description']} with focus on solving specific computational challenges in the target domain.",
        "EXISTING_PROBLEMS": "Current systems in this domain often suffer from limitations in scalability, performance, or feature completeness that impact user experience and system efficiency.",
        "PROPOSED_SOLUTION": f"This {repo_data['name']} project addresses these limitations through modern {primary_language} implementation with improved architecture and user-centric design.",
        "KEY_TECHNOLOGIES": list(repo_data['languages'].keys())[:5],
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

def generate_ieee_paper_content(repo_data, analysis, vector_db, author_name, institution, target_pages):
    """Generate IEEE-formatted research paper content using Groq and a vector DB."""
    
    paper_sections = {}
    
    # Generate Title
    title_prompt = f"""
    Generate a professional IEEE-style research paper title for this project:
    
    Project: {repo_data['name']}
    Description: {repo_data['description']}
    Domain: {analysis['TARGET_DOMAIN']}
    Technologies: {', '.join(analysis['KEY_TECHNOLOGIES'][:3])}
    
    Provide only the title, no quotes or extra text.
    """
    title_response = call_groq_llm(title_prompt, max_tokens=100)
    paper_sections['title'] = title_response.strip().replace('"', '').replace("'", "")
    
    # Generate Abstract (IEEE standard: 150-250 words)
    abstract_query = f"High-level summary, purpose, and key features of the '{repo_data['name']}' project, based on the README file."
    retrieved_docs = vector_db.similarity_search(abstract_query, k=2) # Get top 2 relevant chunks
    context_for_llm = "\n\n---\n\n".join([f"Source: `{doc.metadata['source']}`\n```\n{doc.page_content}\n```" for doc in retrieved_docs])

    abstract_prompt = f"""
    Based on the key information and the following context from the repository, write a 200-word IEEE-standard abstract for this research paper:
    
    CONTEXT FROM REPOSITORY:
    {context_for_llm}

    KEY INFORMATION:
    - Title: {paper_sections['title']}
    - Purpose: {analysis['SYSTEM_PURPOSE']}
    - Solution: {analysis['PROPOSED_SOLUTION']}
    - Innovation: {analysis['INNOVATION']}
    - Technologies: {', '.join(analysis['KEY_TECHNOLOGIES'])}
    
    Structure the abstract with clear sections:
    1. Context and problem statement
    2. Objective and methodology
    3. Key results and findings
    4. Conclusion and significance

    Write in continuous paragraphs but with logical flow between sections and in around 200 words.
    Give paragraphs only, no extra heading. Use technical language appropriate for IEEE publications.
    """
    abstract_response = call_groq_llm(abstract_prompt, max_tokens=300)
    paper_sections['abstract'] = clean_and_format_content(abstract_response)
    
    # Generate Introduction (IEEE format: 3-4 paragraphs)
    intro_query = "The project's main goal, the problem it solves, and its overall purpose, as described in the README or main source files."
    retrieved_docs = vector_db.similarity_search(intro_query, k=4)
    context_for_llm = "\n\n---\n\n".join([f"Source: `{doc.metadata['source']}`\n```\n{doc.page_content}\n```" for doc in retrieved_docs])
    
    intro_prompt = f"""
    Based on the project analysis and the following CONTEXT from the repository, write an IEEE-standard Introduction section with clear structure:
    
    CONTEXT FROM REPOSITORY:
    {context_for_llm}

    Project Context:
    - Name: {repo_data['name']}
    - Domain: {analysis['TARGET_DOMAIN']}
    - Purpose: {analysis['SYSTEM_PURPOSE']}
    - Problems: {analysis['EXISTING_PROBLEMS']}
    - Solution: {analysis['PROPOSED_SOLUTION']}

    TASK:
    Write a multi-paragraph Introduction. Use the CONTEXT to provide specific details.
    
    Structure the introduction as follows and use bullet points where appropriate:
    
    PARAGRAPH 1: Context and Background
    - Introduce the application domain
    - Explain current trends and challenges
    - Highlight the importance of the problem
    
    PARAGRAPH 2: Problem Statement
    - Clearly define specific problems in existing systems
    - List key limitations:
      • Performance issues
      • Scalability concerns
      • Feature gaps
      • User experience problems
    
    PARAGRAPH 3: Proposed Approach
    - Present your solution methodology
    - Key technical approaches:
      • Architecture design decisions
      • Technology stack choices
      • Implementation strategies
    
    PARAGRAPH 4: Contributions and Organization
    - Main contributions of this work:
      • Technical innovations
      • Performance improvements
      • Feature enhancements
    - Paper organization overview
    
    Use formal academic language with proper IEEE citation placeholders [1], [2], etc.
    Format lists with bullet points when listing multiple items.
    DO NOT include the section title "Introduction" in your response - only provide the content.
    """
    intro_response = call_groq_llm(intro_prompt, max_tokens=800)
    paper_sections['introduction'] = clean_and_format_content(intro_response)
    
    # Generate Literature Review/Related Work
    related_query = f"What are the core algorithms, libraries, frameworks, or architectural patterns used in this project? Find details about '{', '.join(analysis['KEY_TECHNOLOGIES'])}'."
    retrieved_docs = vector_db.similarity_search(related_query, k=4)
    context_for_llm = "\n\n---\n\n".join([f"Source: `{doc.metadata['source']}`\n```\n{doc.page_content}\n```" for doc in retrieved_docs])

    literature_prompt = f"""
    The following CONTEXT describes the technical foundation of the project.

    CONTEXT FROM REPOSITORY:
    {context_for_llm}

    ---
    TASK:
    Write an IEEE-standard Related Work section for this research:
    
    Project: {repo_data['name']}
    Domain: {analysis['TARGET_DOMAIN']}
    Technologies: {', '.join(analysis['KEY_TECHNOLOGIES'])}
    Innovation: {analysis['INNOVATION']}
    
    Structure with clear subsections:
    
    A. Existing Approaches in {analysis['TARGET_DOMAIN']}
    - Overview of current methodologies
    - Key technologies and frameworks being used
    
    B. Comparative Analysis
    Analyze existing solutions with their limitations:
    • Solution 1 : Strengths and weaknesses
    • Solution 2 : Performance and scalability issues
    • Solution 3 : Feature limitations and gaps
    
    C. Technology Stack Comparison
    Compare different technology approaches:
    • Traditional approaches vs. modern frameworks
    • Performance considerations
    • Scalability factors
    • Maintenance and development efficiency
    
    D. Research Gap Identification
    - Current limitations in existing work
    - Opportunities for improvement
    - How this work addresses identified gaps
    
    Use bullet points and clear subsections for better readability.
    DO NOT include the section title "Related Work" in your response - only provide the content.
    """
    literature_response = call_groq_llm(literature_prompt, max_tokens=700)
    paper_sections['related_work'] = clean_and_format_content(literature_response)
    
    # Generate Methodology/System Design
    methodology_query = "Describe the system architecture, design patterns, core algorithms, and data structures used in this project. Provide code examples."
    
    # Retrieve relevant documents from the vector database
    retrieved_docs = vector_db.similarity_search(methodology_query, k=6) # Get top 6 relevant chunks
    
    # Format the retrieved context for the LLM
    context_for_llm = "\n\n---\n\n".join([f"Relevant chunk from `{doc.metadata['source']}`:\n```\n{doc.page_content}\n```" for doc in retrieved_docs])

    methodology_prompt = f"""
    Based on the following context from the repository's files, write an IEEE-standard Methodology section with clear subsections:
    
    CONTEXT FROM REPOSITORY:
    {context_for_llm}

    TASK:
    Write an IEEE-standard Methodology section with clear subsections. Use the provided context to be specific and accurate.

    Project Details:
    - Name: {repo_data['name']}
    - Architecture: {analysis['ARCHITECTURE']}
    - Technologies: {', '.join(analysis['KEY_TECHNOLOGIES'])}
    - Methodology: {analysis['METHODOLOGY']}
    - Key Features: {', '.join(analysis['KEY_FEATURES'])}
    
    Repository Information:
    - Files: {len(repo_data['repo_structure'])} total files
    - Languages: {list(repo_data['languages'].keys())}
    - Size: {repo_data['size']} KB
    
    Structure with clear subsections:
    
    A. System Architecture Overview
    - High-level system design
    - Component interaction and data flow
    - Architectural patterns employed
    
    B. Technology Stack and Framework Selection
    Primary Technologies:
    • {analysis['KEY_TECHNOLOGIES'][0] if analysis['KEY_TECHNOLOGIES'] else 'Core Technology'}: Core functionality implementation
    • {analysis['KEY_TECHNOLOGIES'][1] if len(analysis['KEY_TECHNOLOGIES']) > 1 else 'Supporting Technology'}: Supporting framework
    • {analysis['KEY_TECHNOLOGIES'][2] if len(analysis['KEY_TECHNOLOGIES']) > 2 else 'Additional Technology'}: Additional features
    
    Technology Selection Criteria:
    • Performance requirements
    • Scalability considerations
    • Development efficiency
    • Community support and documentation
    
    C. Development Methodology
    - Agile development approach
    - Version control and collaboration
    - Testing and quality assurance strategy
    
    D. Key Algorithms and Data Structures
    Core Implementation Components:
    • Data processing algorithms
    • User interface design patterns
    • Database design and optimization
    • Security and authentication mechanisms
    
    E. System Integration and Deployment
    - Integration testing approach
    - Deployment strategy and environment setup
    - Performance monitoring and optimization
    
    Write 500-600 words with technical details and clear subsection formatting.
    DO NOT include the section title "Methodology" in your response - only provide the content.
    """
    methodology_response = call_groq_llm(methodology_prompt, max_tokens=800)
    paper_sections['methodology'] = clean_and_format_content(methodology_response)
    
    # Generate Implementation Details
    implementation_query = f"Code snippets showing the implementation of key features like '{', '.join(analysis['KEY_FEATURES'])}'. Also find setup, configuration files, and comments about technical challenges."
    retrieved_docs = vector_db.similarity_search(implementation_query, k=6)
    context_for_llm = "\n\n---\n\n".join([f"Source: `{doc.metadata['source']}`\n```\n{doc.page_content}\n```" for doc in retrieved_docs])

    implementation_prompt = f"""
    Based on the following specific code snippets and file contents from the repository, write an Implementation section with detailed technical breakdown:
    
    Technical Details:
    - Primary Language: {list(repo_data['languages'].keys())[0] if repo_data['languages'] else 'Multiple'}
    - Technologies: {', '.join(analysis['KEY_TECHNOLOGIES'])}
    - Architecture: {analysis['ARCHITECTURE']}
    - Features: {', '.join(analysis['KEY_FEATURES'])}
    - Performance: {analysis['PERFORMANCE_CONSIDERATIONS']}

    CONTEXT FROM REPOSITORY:
    {context_for_llm}

    ---
    TASK:
    Structure with clear subsections:
    
    A. Development Environment and Setup
    - IDE and development tools used
    - Project structure and organization
    - Dependency management and build system
    
    B. Core Implementation Components
    Key Modules Implemented:
    • {analysis['KEY_FEATURES'][0] if analysis['KEY_FEATURES'] else 'Core Module'}: Primary functionality
    • {analysis['KEY_FEATURES'][1] if len(analysis['KEY_FEATURES']) > 1 else 'Secondary Module'}: Supporting features
    • {analysis['KEY_FEATURES'][2] if len(analysis['KEY_FEATURES']) > 2 else 'Additional Module'}: Enhanced capabilities
    
    C. Technical Implementation Challenges
    Major challenges addressed:
    • Performance optimization requirements
    • Cross-platform compatibility
    • User interface responsiveness
    • Data management and persistence
    • Security implementation
    
    D. Code Organization and Architecture
    - Modular design principles
    - Separation of concerns
    - Design patterns implementation
    - Code reusability and maintainability
    
    E. Quality Assurance and Testing
    Testing Strategy:
    • Unit testing for individual components
    • Integration testing for system functionality
    • User acceptance testing
    • Performance and load testing
    
    Repository Stats:
    - Total Files: {len(repo_data['repo_structure'])}
    - Repository Size: {repo_data['size']} KB
    - Contributors: Based on commit history
    - License: {repo_data['license']}
    
    Write 400-500 words with specific technical details and clear subsections.
    DO NOT include the section title "Implementation" in your response - only provide the content.
    """
    implementation_response = call_groq_llm(implementation_prompt, max_tokens=700)
    paper_sections['implementation'] = clean_and_format_content(implementation_response)
    
    # Generate Results and Evaluation
    results_query = "Find evidence of functionality, such as test files, testing scripts, benchmark results, or example outputs. Look for quantitative data."
    retrieved_docs = vector_db.similarity_search(results_query, k=5)
    context_for_llm = "\n\n---\n\n".join([f"Source: `{doc.metadata['source']}`\n```\n{doc.page_content}\n```" for doc in retrieved_docs])

    results_prompt = f"""
    Based on the project's public metrics and any evidence of testing or performance found in the CONTEXT, write a Results and Evaluation section with quantitative and qualitative analysis:
    
    CONTEXT FROM REPOSITORY (e.g., test files, benchmarks):
    {context_for_llm}

    Project Metrics:
    - Repository Stars: {repo_data['stars']}
    - Forks: {repo_data['forks']}
    - Key Features: {', '.join(analysis['KEY_FEATURES'])}
    - Scalability: {analysis['SCALABILITY']}
    
    TASK:
    Write a Results and Evaluation section.
    Structure with clear evaluation criteria:
    
    A. Implementation Results
    Successfully implemented features:
    • {analysis['KEY_FEATURES'][0] if analysis['KEY_FEATURES'] else 'Primary Feature'}: Core functionality delivered
    • {analysis['KEY_FEATURES'][1] if len(analysis['KEY_FEATURES']) > 1 else 'Secondary Feature'}: Additional capabilities
    • {analysis['KEY_FEATURES'][2] if len(analysis['KEY_FEATURES']) > 2 else 'Enhanced Feature'}: Advanced functionality
    
    B. Performance Evaluation
    System Performance Metrics:
    • Response time and throughput
    • Resource utilization efficiency
    • Scalability under load
    • Memory and processing optimization
    
    C. User Adoption and Community Response
    Community Metrics:
    • GitHub Stars: {repo_data['stars']} (indicates user interest)
    • Repository Forks: {repo_data['forks']} (shows developer engagement)
    • Commit Activity: {len(repo_data['commits'])} recent commits analyzed
    • Code Quality: Well-structured with {len(repo_data['repo_structure'])} organized files
    
    D. Functional Validation
    Feature Completeness Assessment:
    • Core functionality implementation: Complete
    • User interface and experience: Implemented
    • System integration: Functional
    • Error handling and robustness: Addressed
    
    E. Comparative Analysis
    Advantages over existing solutions:
    • Improved performance characteristics
    • Enhanced user experience design
    • Modern technology stack utilization
    • Better maintainability and extensibility
    
    Write 400-500 words with quantitative and qualitative results.
    DO NOT include the section title "Results and Evaluation" in your response - only provide the content.
    """
    results_response = call_groq_llm(results_prompt, max_tokens=700)
    paper_sections['results'] = clean_and_format_content(results_response)
    
    # Generate Discussion
    discussion_query = "Code comments that mention 'TODO', 'FIXME', 'NOTE', 'limitation', or 'future work'. Also, find the most complex or innovative algorithm implementations."
    retrieved_docs = vector_db.similarity_search(discussion_query, k=5)
    context_for_llm = "\n\n---\n\n".join([f"Source: `{doc.metadata['source']}`\n```\n{doc.page_content}\n```" for doc in retrieved_docs])

    discussion_prompt = f"""
    Based on the project's innovative aspects and any limitations or future plans found in the CONTEXT, write an IEEE-standard Discussion section with critical analysis:
    
    CONTEXT FROM REPOSITORY (comments, complex code):
    {context_for_llm}

    ANALYSIS :
    - Project: {repo_data['name']}
    - Innovation: {analysis['INNOVATION']}
    - Challenges: {', '.join(analysis['TECHNICAL_CHALLENGES'])}
    - Domain Impact: {analysis['TARGET_DOMAIN']}

    TASK:
    Write a critical Discussion section.
    Structure with analytical depth:
    
    A. Significance of Results
    - Technical contributions achieved
    - Performance improvements demonstrated
    - Feature completeness and functionality
    
    B. Implications for {analysis['TARGET_DOMAIN']} Domain
    Impact Areas:
    • Development efficiency improvements
    • User experience enhancements
    • Technology adoption and trends
    • Community and ecosystem benefits
    
    C. Advantages and Benefits
    Key Strengths:
    • {analysis['INNOVATION'][:100]}...
    • Modern architecture and design patterns
    • Scalability and performance optimization
    • Maintainability and code quality
    
    D. Limitations and Areas for Improvement
    Current Limitations:
    • Platform-specific constraints
    • Scalability boundaries
    • Feature scope limitations
    • Performance optimization opportunities
    
    E. Lessons Learned
    Development Insights:
    • Technology selection decisions
    • Implementation approach effectiveness
    • Testing and validation strategies
    • Community feedback integration
    
    F. Future Research Directions
    Potential Enhancements:
    • Advanced feature implementations
    • Performance optimization strategies
    • Cross-platform compatibility
    • Integration with emerging technologies
    • Scalability improvements
    
    Write 500-600 words with critical analysis and technical depth.
    DO NOT include the section title "Discussion" in your response - only provide the content.
    """
    discussion_response = call_groq_llm(discussion_prompt, max_tokens=800)
    paper_sections['discussion'] = clean_and_format_content(discussion_response)
    
    # Generate Conclusion
    conclusion_query = f"A concise summary of the '{repo_data['name']}' project's purpose and main achievement from the README or abstract."
    retrieved_docs = vector_db.similarity_search(conclusion_query, k=2)
    context_for_llm = "\n\n---\n\n".join([f"Source: `{doc.metadata['source']}`\n```\n{doc.page_content}\n```" for doc in retrieved_docs])

    conclusion_prompt = f"""
    Based on the project summary and the following high-level CONTEXT, write a concise IEEE-standard Conclusion.
    
    CONTEXT FROM REPOSITORY:
    {context_for_llm}

    Summarize:
    - Problem: {analysis['EXISTING_PROBLEMS']}
    - Solution: {analysis['PROPOSED_SOLUTION']}
    - Key Contributions: {analysis['INNOVATION']}
    - Results: Successful implementation with {repo_data['stars']} stars
    
    TASK:
    Write a 200-300 word Conclusion.
    Structure the conclusion clearly:
    
    A. Problem Summary and Solution Overview
    - Restate the core problem addressed
    - Summarize the proposed solution approach
    
    B. Key Technical Contributions
    Primary Achievements:
    • Successful implementation of {repo_data['name']}
    • Integration of modern {', '.join(analysis['KEY_TECHNOLOGIES'][:3])} technologies
    • {analysis['INNOVATION'][:80]}...
    • Community adoption with {repo_data['stars']} stars and {repo_data['forks']} forks
    
    C. Results and Impact
    Measurable Outcomes:
    • Functional system implementation
    • Positive community response and adoption
    • Demonstrated performance and reliability
    • Contribution to {analysis['TARGET_DOMAIN']} domain
    
    D. Future Work and Research Directions
    Planned Enhancements:
    • Feature expansion and optimization
    • Performance and scalability improvements
    • Community feedback integration
    • Technology stack evolution
    
    E. Final Impact Statement
    - Contribution to the field
    - Benefits for users and developers
    - Advancement of state-of-the-art
    
    Write in conclusive, authoritative tone appropriate for IEEE publication (200-300 words).
    DO NOT include the section title "Conclusion" in your response - only provide the content.
    """
    conclusion_response = call_groq_llm(conclusion_prompt, max_tokens=400)
    paper_sections['conclusion'] = clean_and_format_content(conclusion_response)
    
    # Generate IEEE-style References
    references_prompt = f"""
    Generate 12-15 IEEE-style references for a paper about:
    
    Topic: {paper_sections['title']}
    Domain: {analysis['TARGET_DOMAIN']}
    Technologies: {', '.join(analysis['KEY_TECHNOLOGIES'])}
    
    Include mix of:
    - Journal articles (IEEE, ACM, Springer)
    - Conference papers (recent, relevant conferences)
    - Technical books
    - Online documentation (for technologies used)
    - Standards and specifications
    
    Format in strict IEEE reference style:
    [1] A. Author, "Title of paper," IEEE Trans. Technology, vol. X, no. Y, pp. ZZ-ZZ, Month Year.
    [2] B. Author et al., "Conference paper title," in Proc. Conference Name, City, Country, Year, pp. XX-XX.
    
    Make them realistic and relevant to the specific domain and technologies.
    """
    references_response = call_groq_llm(references_prompt, max_tokens=800)
    paper_sections['references'] = clean_and_format_content(references_response)
    
    return paper_sections

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
        ("I. INTRODUCTION", paper_sections['introduction']),
        ("II. RELATED WORK", paper_sections['related_work']),
        ("III. METHODOLOGY", paper_sections['methodology']),
        ("IV. IMPLEMENTATION", paper_sections['implementation']),
        ("V. RESULTS AND EVALUATION", paper_sections['results']),
        ("VI. DISCUSSION", paper_sections['discussion']),
        ("VII. CONCLUSION", paper_sections['conclusion'])
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
    refs_elements = process_content_with_formatting(paper_sections['references'], body_style)
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