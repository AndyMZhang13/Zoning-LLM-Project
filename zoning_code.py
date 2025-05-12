import fitz  # PyMuPDF
import re
import faiss
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# Step 1: PDF text extraction
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    all_text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text()
        all_text += f"\n--- Page {page_num + 1} ---\n{text}"
    doc.close()
    return all_text

# Step 2: Clean text
def clean_text(text):
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()

# Step 3: Split into ~200-word chunks
def split_into_chunks(text, chunk_size=200, overlap=0):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

# Step 4: Embed with sentence-transformers
def build_faiss_index(chunks, model_name='all-MiniLM-L6-v2'):
    print(f"[*] Loading model: {model_name}")
    model = SentenceTransformer(model_name)

    print("[*] Generating embeddings...")
    embeddings = model.encode(chunks, show_progress_bar=True, convert_to_numpy=True)

    print("[*] Building FAISS index...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    return index, embeddings

# Step 5: Save index and chunks
def save_chunks_and_index(chunks, index, embeddings, index_path="zoning.index", meta_path="zoning_chunks.txt"):
    faiss.write_index(index, index_path)
    np.save("embeddings.npy", embeddings)

    with open(meta_path, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks):
            f.write(f"--- Chunk {i+1} ---\n{chunk}\n\n")

    print(f"[*] Saved {len(chunks)} chunks, FAISS index, and embeddings.")

# Main script
if __name__ == "__main__":
    pdf_path = "C:/Users/EXAMPLEUSER/SOMEFOLDER/test.pdf" 

    print("[*] Extracting and processing text...")
    raw_text = extract_text_from_pdf(pdf_path)
    print("read text")
    cleaned_text = clean_text(raw_text)
    print("cleaned text")
    chunks = split_into_chunks(cleaned_text, chunk_size=200)
    print("chunked")
    index, embeddings = build_faiss_index(chunks)
    print("embedded")
    save_chunks_and_index(chunks, index, embeddings)
    print("saved embeddings")

    print("DONE")
