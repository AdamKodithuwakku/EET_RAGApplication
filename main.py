import os, glob, json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI
from typing import Optional


# Creating the FastAPI app
app = FastAPI()


documents = {
    "uom-about.txt": ["General", "2025"],
    "uom-ee-about.txt" : ["EE", "2025", "Electrical", "Electrical Engineering"],
    "uom-cse-about.txt" : ["CSE", "2025", "Computer", "Computer Science" ]
}


# Creating the FastAPI app
app = FastAPI()

# Define a root endpoint
@app.get("/")
async def read_root():
    return {"message": "API is running"}


# Define an endpoint with a query parameter
@app.get("/search/")
async def search_admissions(
    query: str,
    department: Optional[str] = None,
    Year: Optional[int] = None
):
    
    # Initialize the model, index, and chunked corpus
    model, index, chunked_corp = model_init()  
    query_embedding = model.encode([query], convert_to_numpy=True)
    query_vec = normalize(query_embedding).astype('float32')

    
    # Retrieve the nth most compatible from the index 
    result = index.search(query_vec, 2) 

    text, message = pretty_result(result, chunked_corp, department, Year)
    message["Supplemental"] = text
    return message


def pretty_result(result, chunked_corp, departemnt = None, year = "2025"):
    """Pretty prints the search results."""
    distances, indices = result

    text = "Top Related Results:\n"
    message = {}

    count  = 0
    for dist, idx in zip(distances[0], indices[0]):
        chunk = chunked_corp[idx]

        if not departemnt or departemnt and departemnt in documents[chunk['doc_id']]:
            message[count] = chunk['text']
        else:
            text += f"Document ID: {chunk['doc_id']}, Chunk ID: {chunk['chunk_id']}, Similarity: {dist:.4f}\n"
            text += f"Text: {chunk['text']}\n\n"

        count += 1

    return text,  message


def model_init():
    """Initializes the model, index, and chunked corpus."""
    corpus = {}
    for doc in documents.keys():
        with open(os.path.join("docs", doc), 'r', encoding='utf-8') as file:
            corpus[doc] = file.read()
    
    chunked_corp = chunked_corpus(corpus, chunk_size=20, overlap=5)


    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode([chunk['text'] for chunk in chunked_corp], show_progress_bar=True, convert_to_numpy=True)


    vec = normalize(embeddings).astype('float32')
    dimension = vec.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(vec)


    return model, index, chunked_corp


def chunked_corpus(corpus, chunk_size=500, overlap=50):
    """Creates chunked corpus from the input corpus."""
    chunked_corpus = []
    for doc_id, text in corpus.items():
        chunks = chunk_text(text, chunk_size, overlap)
        for i, chunk in enumerate(chunks):
            chunked_corpus.append({
                'doc_id': doc_id,
                'chunk_id': i,
                'text': chunk
            })

    return chunked_corpus
    

def chunk_text(text, chunk_size=500, overlap=50):
    """Splits the input text into chunks of specified size with overlap."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks


def normalize(vec):
    """Normalize the vectors to unit."""
    norm = np.linalg.norm(vec, axis=1, keepdims=True)
    return vec / norm
