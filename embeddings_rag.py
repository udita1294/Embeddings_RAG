import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
import numpy as np
from sentence_transformers import SentenceTransformer

def cosine_similarity(a,b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

model = SentenceTransformer("all-MiniLM-L6-v2")
text = "Machine Learning is fun"

embedding = model.encode(text)
# print(embedding.shape)  # Output: (384)
# print(embedding[:10])  # Output: [0.12345678 0.23456789 ...] (example values)

t1 = "There are 24 paid leaves"
t2 = "There are 24 vacation days"

v1 = model.encode(t1)
v2 = model.encode(t2)

print("Cosine similarity between t1 and t2:", cosine_similarity(v1, v2))
