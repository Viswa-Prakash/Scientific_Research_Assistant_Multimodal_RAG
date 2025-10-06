# rag.py
import os
import io
import numpy as np
import torch
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from PIL import Image

from transformers import AutoTokenizer, AutoModel, CLIPProcessor, CLIPModel
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import init_chat_model
from langchain.schema.messages import HumanMessage

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import faiss

# -----------------------------
# Environment
# -----------------------------
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# SciBERT for text embeddings
# -----------------------------
tokenizer_sci = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
model_sci = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased").to(device)
model_sci.eval()

@torch.no_grad()
def embed_text_sci(text: str) -> np.ndarray:
    """Embed text using SciBERT; return 768-dim normalized vector."""
    if not text:
        text = ""
    inputs = tokenizer_sci(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    outputs = model_sci(**inputs)
    # Mean pooling of last hidden state
    last_hidden = outputs.last_hidden_state  # (1, seq_len, 768)
    mask = inputs['attention_mask'].unsqueeze(-1)
    mean_emb = (last_hidden * mask).sum(1) / mask.sum(1)
    mean_emb = mean_emb / mean_emb.norm(dim=-1, keepdim=True)
    return mean_emb.cpu().numpy()[0].astype(np.float32)

# -----------------------------
# CLIP for images + multimodal text → image retrieval
# -----------------------------
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()

@torch.no_grad()
def embed_image_clip(image: Image.Image) -> np.ndarray:
    """Embed PIL image using CLIP image encoder (512-dim)."""
    inputs = clip_processor(images=[image], return_tensors="pt").to(device)
    feats = clip_model.get_image_features(**inputs)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy()[0].astype(np.float32)

@torch.no_grad()
def embed_text_clip(text: str) -> np.ndarray:
    """Embed text using CLIP text encoder (512-dim) for text→image retrieval."""
    if not text:
        text = ""
    inputs = clip_processor(text=[text], return_tensors="pt", padding=True, truncation=True, max_length=77).to(device)
    feats = clip_model.get_text_features(**inputs)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy()[0].astype(np.float32)

# -----------------------------
# FAISS VectorStore (dimension-safe)
# -----------------------------
class VectorStore:
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.docs = []

    def add(self, emb: np.ndarray, doc: Document):
        emb = np.asarray(emb, dtype=np.float32)
        if emb.ndim == 1:
            emb = emb.reshape(1, -1)
        self.index.add(emb)
        self.docs.append(doc)

    def search(self, query_emb: np.ndarray, k: int = 5):
        query_emb = np.asarray(query_emb, dtype=np.float32)
        if query_emb.ndim == 1:
            query_emb = query_emb.reshape(1, -1)
        if query_emb.shape[1] != self.dimension:
            raise ValueError(f"Query dimension {query_emb.shape[1]} does not match index dimension {self.dimension}")
        
        if len(self.docs) == 0:
            return []  # safely return empty list if no documents

        D, I = self.index.search(query_emb, min(k, len(self.docs)))
        results = [self.docs[i] for i in I[0] if i != -1]
        return results


    def __len__(self):
        return len(self.docs)


# -----------------------------
# Scientific Research Assistant RAG
# -----------------------------
class ScientificResearchRAG:
    def __init__(self, text_dim: int = 768, img_dim: int = 512):
        self.vstore_text = VectorStore(text_dim)
        self.vstore_img = VectorStore(img_dim)
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=100)
        self.llm = init_chat_model("openai:gpt-4.1")
        self.checkpointer = MemorySaver()
        self.image_store = {}  # {name: PIL.Image}
        self.graph = self._create_graph()

    # -------------------------
    # Ingest PDF bytes (text + images)
    # -------------------------
    def ingest_pdf_bytes(self, pdf_bytes: bytes, source_name: str):
        reader = PdfReader(io.BytesIO(pdf_bytes))
        for i, page in enumerate(reader.pages):
            try:
                txt = page.extract_text() or ""
            except Exception:
                txt = ""
            if txt.strip():
                doc = Document(page_content=txt, metadata={"source": source_name, "page": i, "type": "text"})
                chunks = self.splitter.split_documents([doc])
                for chunk in chunks:
                    emb = embed_text_sci(chunk.page_content)
                    self.vstore_text.add(emb, Document(page_content=chunk.page_content, metadata=chunk.metadata))
            # Optional: extract images (figures) from page
            # Note: PyPDF2 cannot extract images directly; user can provide separately
            # For simplicity, user can call ingest_figures() to add images

    def ingest_figures(self, images: list):
        """
        images: list of {"image": PIL.Image, "name": str, "source": str (optional)}
        """
        for img_data in images:
            img = img_data["image"]
            name = img_data.get("name", f"figure_{len(self.image_store)+1}")
            emb = embed_image_clip(img)
            doc = Document(page_content=f"Image: {name}", metadata={"source": img_data.get("source", name), "type": "image"})
            self.vstore_img.add(emb, doc)
            self.image_store[name] = img.copy()

    # -------------------------
    # LangGraph workflow
    # -------------------------
    def _create_graph(self):
        workflow = StateGraph(dict)

        def retrieve_papers(state):
            q_text = state.get("query_text", "")
            if not q_text:
                return {"query_text": q_text, "papers": [], "figures": []}
            q_emb = embed_text_sci(q_text)
            hits = self.vstore_text.search(q_emb, k=12)
            return {"query_text": q_text, "papers": hits, "figures": []}

        def retrieve_figures(state):
            q_text = state.get("query_text", "")
            # Use CLIP text embeddings to query image FAISS (dimension-safe)
            q_emb = embed_text_clip(q_text)
            hits_img = self.vstore_img.search(q_emb, k=6)
            return {"query_text": q_text, "papers": state.get("papers", []), "figures": hits_img}

        def summarize_findings(state):
            q_text = state.get("query_text", "")
            papers = state.get("papers", []) or []
            figures = state.get("figures", []) or []

            context = "\n\n".join([f"[{d.metadata.get('source','')}] {d.page_content[:800]}" for d in papers[:6]])
            prompt_parts = [{"type": "text", "text": f"User question: {q_text}\n\n"}]
            if context:
                prompt_parts.append({"type": "text", "text": "Relevant paper excerpts:\n" + context + "\n\n"})
            if figures:
                figure_names = ", ".join([f.metadata.get("source") for f in figures[:6]])
                prompt_parts.append({"type": "text", "text": f"Relevant figures: {figure_names}\n\n"})
            prompt_parts.append({"type": "text", "text": "Provide a concise summary with insights, references, and explanations of figures."})

            message = HumanMessage(content=prompt_parts)
            try:
                response = self.llm.invoke([message])
                answer_text = getattr(response, "content", str(response))
            except Exception as e:
                answer_text = f"LLM call failed: {e}"

            # Include figure images for display
            images_out = []
            for d in figures:
                src = d.metadata.get("source")
                if src and src in self.image_store:
                    images_out.append({"name": src, "image": self.image_store[src]})

            return {"answer": {"text": answer_text, "images": images_out}}

        workflow.add_node("retrieve_papers", retrieve_papers)
        workflow.add_node("retrieve_figures", retrieve_figures)
        workflow.add_node("summarize_findings", summarize_findings)

        workflow.set_entry_point("retrieve_papers")
        workflow.add_edge("retrieve_papers", "retrieve_figures")
        workflow.add_edge("retrieve_figures", "summarize_findings")
        workflow.add_edge("summarize_findings", END)

        return workflow.compile(checkpointer=self.checkpointer)

    # -------------------------
    # Query
    # -------------------------
    def query(self, query_text: str):
        if not query_text or not query_text.strip():
            return {"text": "Please provide a question.", "images": []}
        state = {"query_text": query_text}
        res = self.graph.invoke(state, config={"configurable": {"thread_id": "streamlit-session"}})
        ans = res.get("answer") if isinstance(res, dict) else res
        if not ans:
            return {"text": "No answer found.", "images": []}
        return {"text": ans.get("text", ""), "images": ans.get("images", [])}
