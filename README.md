# Sentence Embedding & Cosine Similarity

A simple Python project that demonstrates how to convert text into **vector embeddings** using Sentence Transformers and measure the **semantic similarity** between two sentences using **cosine similarity**.

## Overview

This project uses the `all-MiniLM-L6-v2` model from Sentence Transformers to generate numerical representations (embeddings) of sentences.

For example:

* `"There are 24 paid leaves"`
* `"There are 24 vacation days"`

Although these sentences use different words (`leaves` vs `vacation days`), they have a similar meaning. Sentence embeddings allow us to capture this semantic relationship.

The project then calculates the **cosine similarity** between their embeddings.

## Technologies Used

* **Python**
* **NumPy**
* **Sentence Transformers**
* **all-MiniLM-L6-v2**
* **scikit-learn concepts / cosine similarity**
* **dotenv** (included for environment configuration)

## How It Works

The process consists of three main steps:

### 1. Convert text into embeddings

The Sentence Transformer model converts a sentence into a numerical vector.

```python
model = SentenceTransformer("all-MiniLM-L6-v2")

text = "Machine Learning is fun"
embedding = model.encode(text)
```

The `all-MiniLM-L6-v2` model generates an embedding with **384 dimensions**.

```text
Input:
"There are 24 paid leaves"

Output:
[0.123..., -0.045..., 0.231..., ...]
```

### 2. Generate embeddings for two sentences

```python
t1 = "There are 24 paid leaves"
t2 = "There are 24 vacation days"

v1 = model.encode(t1)
v2 = model.encode(t2)
```

### 3. Calculate cosine similarity

Cosine similarity measures the angle between two vectors.

```python
def cosine_similarity(a, b):
    return np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )
```

The result generally lies between **-1 and 1**, although sentence embeddings commonly produce values in the positive range.

* **1.0** → Very similar
* **0.0** → Little/no similarity
* **-1.0** → Opposite direction


## Understanding Embeddings

An embedding is a numerical representation of text.

For example:

```text
"Machine Learning is fun"
        ↓
Sentence Transformer
        ↓
384-dimensional vector
        ↓
[0.12, -0.04, 0.31, ..., 0.08]
```

Instead of comparing sentences based only on exact words, embeddings allow us to compare their **meaning**.

This is an important concept behind many modern AI applications such as:

* Semantic search
* Recommendation systems
* Document similarity
* Question answering
* Retrieval-Augmented Generation (RAG)
* Duplicate detection
* Text clustering

## Why Cosine Similarity?

Cosine similarity compares two vectors based on their direction rather than their magnitude.

The formula is:

```text
             A · B
cos(θ) = ─────────────
         ||A|| × ||B||
```

Where:

* `A` = embedding of the first sentence
* `B` = embedding of the second sentence
* `A · B` = dot product
* `||A||` = magnitude of vector A
* `||B||` = magnitude of vector B

