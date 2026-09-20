import os
from pypdf import PdfReader
from openai import OpenAI
import chromadb
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

# 1. Extract text from PDF
reader = PdfReader("data/agent_as_a_judge.pdf")  # match your actual filename
full_text = ""
for page in reader.pages:
    full_text += page.extract_text() + "\n"

# 2. Chunk the text (simple fixed-size chunking with overlap)
def chunk_text(text, chunk_size=1000, overlap=150):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

chunks = chunk_text(full_text)
print(f"Created {len(chunks)} chunks")

# 3. Embed each chunk
def get_embedding(text):
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return resp.data[0].embedding

# 4. Store in ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="agent_as_a_judge")

for i, chunk in enumerate(chunks):
    embedding = get_embedding(chunk)
    collection.add(
        ids=[f"chunk_{i}"],
        embeddings=[embedding],
        documents=[chunk]
    )
    print(f"Embedded chunk {i+1}/{len(chunks)}")

print("Ingestion complete.")