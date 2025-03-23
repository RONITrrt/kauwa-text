import spacy

nlp = spacy.load('en_core_web_sm')

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

def process_news_report(file_path):
    """Reads a text file, extracts claims, and saves them to claims.txt."""
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()

    claims = extract_claims(text)

    with open("claims.txt", "w", encoding="utf-8") as out_file:
        for claim in claims:
            out_file.write(claim + "\n")

    print(f"Extracted {len(claims)} claims. Saved to claims.txt")

# Example usage
process_news_report("news_report.txt")
