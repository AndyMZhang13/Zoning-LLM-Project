import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

def load_index_and_chunks(index_path="zoning.index", meta_path="zoning_chunks.txt"):
    print("[*] Loading FAISS index...")
    index = faiss.read_index(index_path)

    print("[*] Loading text chunks...")
    with open(meta_path, "r", encoding="utf-8") as f:
        raw_chunks = f.read().split("\n\n")
        chunks = [chunk.partition("\n")[2] for chunk in raw_chunks if chunk.strip()]

    return index, chunks

def search(query, index, chunks, model, top_k=5):
    query_embedding = model.encode([query])[0].astype("float32")
    D, I = index.search(np.array([query_embedding]), top_k)

    results = []
    for score, idx in zip(D[0], I[0]):
        if idx < len(chunks):
            results.append((score, chunks[idx]))
    return results

if __name__ == "__main__":
    model = SentenceTransformer("all-MiniLM-L6-v2")
    index, chunks = load_index_and_chunks()

    print("\nReady to search! Type your query below.")
    while True:
        query = input("\n> ")
        if query.lower() in ("exit", "quit"):
            break
        results = search(query, index, chunks, model, top_k=10)
        _best_score = None
        _range = None
        for i, (score, chunk) in enumerate(results, 1):
            #HARD CODING THIS TEMP
            if i == 1:
                _best_score = score
            if i == 10:
                _range = score - _best_score
            # print(f"\n--- Result {i} (distance={score:.4f}) ---\n{chunk}")
        print(f"\nbest: {_best_score}  |  range: {_range}")

        #Unsure but <1.0 distance are very good, and >1.4 are pretty bad