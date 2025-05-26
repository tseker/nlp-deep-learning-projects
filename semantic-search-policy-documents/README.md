
# Semantic Search on Policy Documents using Transformers

This project demonstrates a semantic search pipeline for short policy-related documents using the `sentence-transformers` library. It shows how transformer-based sentence embeddings can retrieve contextually relevant results, even when exact keywords are not present.

##  Key Concepts

- **Sentence Embeddings**: Representing entire sentences as high-dimensional vectors.
- **Cosine Similarity**: Measuring semantic closeness between the query and documents.
- **Transformer Models**: Context-aware encoding using `all-MiniLM-L6-v2`.

##  Use Case

Imagine you are analyzing thousands of policy memos or historical government records. This tool can help you retrieve documents **based on meaning**, not just exact words.

##  Project Files

- `Semantic_Search_Tuba.ipynb` – The full Jupyter notebook with code and explanations.
- `Semantic_Search_Detailed_EN.md` / `.pdf` – English report with explanations and outputs.


##  Technologies Used

- Python
- sentence-transformers (MiniLM)
- PyTorch
- Pandas
- Jupyter Notebook

##  Example Query

> tax reform during financial crisis

Returns:
- "Corporate taxes were adjusted in response to the financial crisis of 2008."

##  Applications

- Legal text retrieval
- Policy document classification
- Economic research search tools
- Semantic search prototypes for institutions

## Author

Created by [Tuba Seker] as part of her NLP & Deep Learning learning projects.
