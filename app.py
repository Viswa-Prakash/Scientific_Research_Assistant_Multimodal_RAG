import streamlit as st
from io import BytesIO
from rag import ScientificResearchRAG
from PyPDF2 import PdfReader
import fitz  # PyMuPDF for images/tables extraction


st.set_page_config(page_title="Scientific Research Assistant", layout="wide")
st.title("Scientific Research Assistant – Multimodal RAG")

# Initialize RAG instance
rag = ScientificResearchRAG()

# --- PDF Upload ---
uploaded_files = st.file_uploader(
    "Upload PDF files (papers with text, figures, tables)", type=["pdf"], accept_multiple_files=True
)

if uploaded_files:
    for pdf_file in uploaded_files:
        pdf_bytes = pdf_file.read()
        rag.ingest_pdf_bytes(pdf_bytes, source_name=pdf_file.name)
    st.success(f"Ingested {len(uploaded_files)} PDF(s) successfully!")

# --- Query ---
st.subheader("Ask a scientific question")
query_text = st.text_area("Enter your question here")

if st.button("Get Answer"):
    if not query_text.strip():
        st.warning("Please enter a question first.")
    else:
        with st.spinner("Retrieving relevant papers and figures..."):
            result = rag.query(query_text)
        st.subheader("Answer")
        st.write(result.get("text", "No answer found."))

        # Display retrieved figures if any
        images = result.get("images", [])
        if images:
            st.subheader("Relevant Figures / Tables")
            for img_data in images:
                st.image(img_data["image"], caption=img_data.get("name", "Figure"), use_column_width=True)
