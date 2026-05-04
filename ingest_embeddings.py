import json
import os
import time
import requests
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

TABLE_NAME = "menu_items"

supabase = create_client(
    os.environ["SUPABASE_URL"],
    os.environ["SUPABASE_SERVICE_KEY"],
)


def embed(texts: list[str]) -> list[list[float]]:
    vectors = []
    for text in texts:
        response = requests.post(
            "http://localhost:11434/api/embeddings",
            json={"model": "nomic-embed-text", "prompt": text}
        )
        vectors.append(response.json()["embedding"])
    return vectors


def ingest(chunks_path: str):
    with open(chunks_path) as f:
        chunks = json.load(f)

    print(f"Loaded {len(chunks)} chunks from {chunks_path}")

    batch_size = 50
    total_upserted = 0

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [c["embedding_text"] for c in batch]

        print(f"Embedding batch {i//batch_size + 1} ({len(batch)} items)...")
        vectors = embed(texts)

        rows = []
        for chunk, vector in zip(batch, vectors):
            rows.append({
                "id":             chunk["id"],
                "restaurant":     chunk["restaurant"],
                "address":        chunk["address"],
                "cuisine":        chunk["cuisine"],
                "category":       chunk["category"],
                "name":           chunk["name"],
                "price":          chunk["price"],
                "spicy":          chunk["spicy"],
                "dietary_tags":   chunk["dietary_tags"],
                "variants":       chunk["variants"],
                "notes":          chunk["notes"],
                "embedding_text": chunk["embedding_text"],
                "embedding":      vector,
            })

        result = (
            supabase.table(TABLE_NAME)
            .upsert(rows, on_conflict="id")
            .execute()
        )

        total_upserted += len(rows)
        print(f"  Upserted {len(rows)} rows. Total so far: {total_upserted}")

        if i + batch_size < len(chunks):
            time.sleep(0.5)

    print(f"\nDone. {total_upserted} items embedded and stored in Supabase.")


if __name__ == "__main__":
    ingest("dolan_rag_chunks.json")