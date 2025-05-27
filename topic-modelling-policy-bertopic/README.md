
# Semantic Search and Topic Modeling on Policy Documents

This project applies **semantic search** and **BERTopic-based topic modeling** to a collection of 200 synthetic policy-related sentences. It is part of a structured NLP and deep learning learning path, designed to build intuition and hands-on experience with text embedding and unsupervised topic extraction.

##  Objectives

- Convert policy sentences into semantic embeddings using `SentenceTransformer`
- Cluster the semantic space using BERTopic (UMAP + HDBSCAN)
- Visualize and interpret discovered topics
- Optionally summarize and label each topic
- Export results for reporting and reproducibility


---

##  Methods Used

- **Embedding**: `all-mpnet-base-v2` via `sentence-transformers`
- **Topic Modeling**: BERTopic with:
  - UMAP (for dimensionality reduction)
  - HDBSCAN (for clustering)
- **Visualizations**: Topic map, bar charts, top words
- **Summarization**: HuggingFace `facebook/bart-large-cnn` (optional)
- **Manual Topic Labeling**: based on top words & representative documents

---

##  Sample Topics Discovered

1. **Education & STEM Investment**
2. **Employment & Job Training**
3. **Subsidies & Energy Tax Reform**
4. **Biotech & Innovation Partnerships**
5. **Housing & Family Support**
6. **International Trade & Sanctions**

---

##  Outputs

- Topic assignments per sentence
- Visualizations of topic structure
- Final report in English and Turkish
- Jupyter Notebook with annotated code

---
##  Note

- For convenience, I recommend using Google colab in this project for the learning purposes.

---

##  Author

Project led by **Tuba Seker** as part of an NLP and Deep Learning projects.
