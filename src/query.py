import os
from openai import OpenAI
import chromadb
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="agent_as_a_judge")

def get_embedding(text):
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return resp.data[0].embedding

def ask(question, k=4):
    q_embedding = get_embedding(question)
    results = collection.query(query_embeddings=[q_embedding], n_results=k)
    retrieved_chunks = results["documents"][0]

    context = "\n\n---\n\n".join(retrieved_chunks)

    prompt = f"""Answer the question using ONLY the context below. If the answer isn't in the context, say so.

Context:
{context}

Question: {question}

Answer:"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    print("\n=== ANSWER ===")
    print(response.choices[0].message.content)
    print("\n=== RETRIEVED CHUNKS ===")
    for i, c in enumerate(retrieved_chunks):
        print(f"\n--- Chunk {i+1} ---\n{c[:300]}...")

if __name__ == "__main__":
    while True:
        q = input("\nAsk a question (or 'quit'): ")
        if q.lower() == "quit":
            break
        ask(q)