"""
BUILD FAISS INDEX FOR PROJECT 
Run this ONCE before genai_metadata_tool.py
"""

import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

# -------------------------
# CONFIG
# -------------------------
VECTORSTORE_PATH = "metadata_index"

# Example script chunks (replace with real script chunks later)
script_chunks = [
    "A group of prisoners plan a dangerous escape from a high-security prison.",
    "The story explores survival, justice, and brotherhood under extreme pressure.",
    "Guards tighten security as tension rises and betrayal is revealed.",
    "Freedom comes at a cost, testing loyalty and sacrifice."
]

# -------------------------
# CREATE DOCUMENTS
# -------------------------
documents = [
    Document(
        page_content=chunk,
        metadata={"similarity": 1.0}
    )
    for chunk in script_chunks
]

# -------------------------
# BUILD VECTORSTORE
# -------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

vectorstore = FAISS.from_documents(documents, embeddings)

# -------------------------
# SAVE TO DISK
# -------------------------
os.makedirs(VECTORSTORE_PATH, exist_ok=True)
vectorstore.save_local(VECTORSTORE_PATH)

print("FAISS index created successfully ✅")
