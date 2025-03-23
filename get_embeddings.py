import os
import pandas as pd
from sentence_transformers import SentenceTransformer

# Load model for embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")


# Read extracted claims
with open("claims.txt", "r", encoding="utf-8") as file:
    claims = file.read().splitlines()
    claims = [claim.strip() for claim in claims if claim.strip()]

print(f"Loaded {len(claims)} claims for embedding.")

# Generate embeddings
embeddings = embedding_model.encode(claims)

# Store embeddings in a DataFrame
df_embeddings = pd.DataFrame({"claim": claims, "embedding": list(embeddings)})
df_embeddings.to_csv("embeddings.csv", index=False)

print("Embeddings saved to embeddings.csv")
