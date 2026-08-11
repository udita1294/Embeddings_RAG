import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
text = "Machine Learning is fun"

embedding = model.encode(text)
print(embedding.shape)  # Output: (384)
print(embedding[:10])  # Output: [0.12345678 0.23456789 ...] (example values)