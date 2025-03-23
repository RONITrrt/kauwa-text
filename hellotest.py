import streamlit as st
import os
import pandas as pd
import spacy
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# Configure Gemini API Key
GOOGLE_API_KEY = "your-api-key-here"  # 🔹 Replace with your actual Gemini API key
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
genai.configure(api_key=GOOGLE_API_KEY)

# Load NLP model
nlp = spacy.load('en_core_web_sm')

# Load sentence embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")


### 📌 Function to Extract Claims ###
def extract_claims(text):
    """Extracts claims from a news report using NLP processing."""
    doc = nlp(text)
    claims = []

    for sent in doc.sents:
        # Look for statements with key indicators of claims
        if any(keyword in sent.text.lower() for keyword in ["suggests", "believes", "states", "argues", "warns", "reports", "announces", "remarks", "concerns", "finds", "reveals", "predicts"]):
            claims.append(sent.text.strip())

        # Capture sentences with named entities (Experts, Organizations)
        elif any(ent.label_ in ["PERSON", "ORG"] for ent in sent.ents):
            claims.append(sent.text.strip())

    return claims


### 📌 Streamlit UI ###
st.title("📰 News Report Claim Analyzer")

# File Upload in Streamlit
uploaded_file = st.file_uploader("Upload a news report (.txt file)", type=["txt"])

if uploaded_file:
    # Read file content
    text = uploaded_file.read().decode("utf-8")

    # Extract claims
    claims = extract_claims(text)

    # Save claims to a file
    with open("claims.txt", "w", encoding="utf-8") as file:
        for claim in claims:
            file.write(claim + "\n")

    # Display success message
    st.success(f"✅ Extracted {len(claims)} claims! You can now download them.")

    # Download Button for `claims.txt`
    with open("claims.txt", "r") as f:
        st.download_button("📥 Download Claims File", f, "claims.txt")

    # Display Extracted Claims
    st.subheader("📌 Extracted Claims:")
    for claim in claims:
        st.write(f"- {claim}")

    # Generate Embeddings
    embeddings = embedding_model.encode(claims)
    df_embeddings = pd.DataFrame({"claim": claims, "embedding": list(embeddings)})
    df_embeddings.to_csv("embeddings.csv", index=False)

    st.success("✅ Embeddings generated and saved!")

    # Chat with Claims
    st.subheader("💬 Ask a Question about the Extracted Claims:")
    query = st.text_input("Enter your query:")

    def get_answer(context, query):
        """Use Gemini AI to answer questions based on extracted claims."""
        prompt = f"Based on the claims below, answer the query:\n\n{context}\n\nQuery: {query}\nAnswer:"
        model = genai.GenerativeModel("gemini-1.5-flash-latest")
        response = model.generate_content(prompt)
        return response.text

    if st.button("Submit Query"):
        if query:
            claims_text = "\n".join(claims)
            answer = get_answer(claims_text, query)
            st.subheader("🧠 Answer:")
            st.write(answer)
        else:
            st.error("⚠️ Please enter a query.")
