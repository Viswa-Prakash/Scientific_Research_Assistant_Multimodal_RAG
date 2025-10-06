# Scientific Research Assistant – Multimodal RAG

A **multimodal Retrieval-Augmented Generation (RAG)** system designed to assist researchers in retrieving and summarizing scientific literature, including **text, figures, and tables**. This tool combines **SciBERT**, **CLIP**, **FAISS**, and **LangGraph** to provide fast, context-aware answers from PDFs and associated figures.

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Technologies](#technologies)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Example](#example)
- [Future Improvements](#future-improvements)

---

## Features

- Upload multiple **PDFs** (research papers, reports, manuals).  
- Extract **text** from PDFs and embed using **SciBERT**.  
- Embed **figures/images** using **CLIP** for multimodal retrieval.  
- Search for relevant papers and figures via **FAISS vector stores**.  
- Summarize findings using **GPT-4.1** via a **LangGraph workflow**.  
- Display **multimodal answers**: text summaries + figures/tables.  
- Modular and extendable for additional modalities or AI models.

---

## Architecture

User Query
│
▼
LangGraph Workflow
├─ retrieve_papers (SciBERT embeddings → FAISS text search)
├─ retrieve_figures (CLIP embeddings → FAISS image search)
└─ summarize_findings (GPT-4.1)
│
▼
Answer + Relevant Figures

---

- **Text Embeddings**: SciBERT (768-dimensional vectors)  
- **Image Embeddings**: CLIP (512-dimensional vectors)  
- **Vector Storage**: FAISS (efficient similarity search)  
- **Workflow Management**: LangGraph (multi-step retrieval)  
- **LLM**: GPT-4.1 for summarization and explanation

---

## Technologies

- **Python 3.10+**  
- **Streamlit** – web interface  
- **PyPDF2** – extract text from PDFs  
- **fitz (PyMuPDF)** – optional figure/table extraction  
- **Transformers** – SciBERT, CLIP  
- **FAISS** – vector similarity search  
- **LangGraph** – multi-step RAG workflow  
- **OpenAI API** – GPT-4.1 for text summarization

---

## Installation

```
1. Clone the repository:

```bash
git clone https://github.com/Viswa-Prakash/Scientific_Research_Assistant_Multimodal_RAG.git
cd Scientific_Research_Assistant_Multimodal_RAG

2. Create a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate     # Windows

3. Install dependencies:

```bash
pip install -r requirements.txt


4. Set up the OpenAI API key:

```bash
export OPENAI_API_KEY="your_openai_api_key"


5. Run the Streamlit app:

```bash
streamlit run app.py
```

---

## Usage

1. Open the Streamlit app in your web browser.
```bash
streamlit run app.py
```
2. Upload PDFs (papers with text, figures, tables).
3. Enter a scientific question in the text area.
4. Click "Get Answer" to retrieve a concise summary with references and associated figures.
5. View relevant figures displayed below the answer.
---