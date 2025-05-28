
# Retrieval-Augmented Generation (RAG) for Policy Question Answering

This project demonstrates how to build a RAG pipeline that combines semantic retrieval using FAISS with answer generation using a transformer-based language model. It uses a dataset of 200 diverse policy statements.


##  Project Overview

1. **Sentence Embedding**  
   - Embedding with `all-MiniLM-L6-v2` from `sentence-transformers`

2. **Semantic Search**  
   - FAISS index creation with dot product on normalized vectors

3. **Answer Generation**  
   - Query-based retrieval
   - Prompt construction
   - Response generation using `flan-t5-base`

4. ** Theoretical Explanation**  
   - Cosine similarity vs. dot product  
   - Importance of normalization  
   - Context length limits and `top_k` retrieval strategy

##  Requirements

```bash
pip install sentence-transformers faiss-cpu transformers
```



##  Tips

- Truncate prompts to avoid hitting LLM token limits
- Always normalize vectors for FAISS cosine search
- Use clearer prompt structure for better answers
- Use rerankers or `top_k < 15` for quality results

---

Built by **Tuba Seker**, May 2025.
