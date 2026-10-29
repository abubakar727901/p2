import numpy as np


def chunk_text(text, chunk_size=900, overlap=150):
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_size must be positive and overlap must be smaller than chunk_size.")
    normalized = " ".join(text.split())
    if not normalized:
        return []
    chunks = []
    start = 0
    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))
        if end < len(normalized):
            boundary = normalized.rfind(" ", start + chunk_size // 2, end)
            if boundary > start:
                end = boundary
        piece = normalized[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(normalized):
            break
        start = max(end - overlap, start + 1)
    return chunks


def create_embeddings(client, texts, model="text-embedding-3-small", batch_size=64):
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    vectors = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        response = client.embeddings.create(model=model, input=batch)
        ordered = sorted(response.data, key=lambda item: item.index)
        vectors.extend(item.embedding for item in ordered)
    result = np.asarray(vectors, dtype=np.float32)
    if result.ndim != 2 or result.shape[0] != len(texts) or not np.isfinite(result).all():
        raise ValueError("The embedding service returned invalid or incomplete vectors.")
    return result
