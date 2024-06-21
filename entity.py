import spacy

# Load spaCy's pre-trained model
nlp = spacy.load("en_core_web_sm")

def extract_entities(text):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

# Example usage
#print(extract_entities("Can you book a flight to New York for tomorrow?"))
