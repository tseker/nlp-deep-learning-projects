
# Token-Level Policy Component Extraction (NER)

This project builds a Named Entity Recognition (NER) model to extract structured policy components from economic policy texts using BERT-based token classification.

## Project Overview

In policy documents, specific elements such as monetary amounts, tools, target groups, or time references often appear as unstructured text. This project transforms these into structured data using a fine-tuned BERT model.

### Entities to Extract:
- amount — numeric values (e.g., "15%", "1 billion dollars")
- policy_tool — instruments used (e.g., "tax", "ban", "subsidy")
- target_group — groups affected (e.g., "consumers", "farmers")
- sector — domain/industry (e.g., "healthcare", "transport")
- region — location or jurisdiction
- time_period — temporal information (e.g., "2026", "next year")

These are labeled using the BIO tagging scheme (Begin, Inside, Outside) at the token level.

## Example

Sentence:  
"A 10% tax on luxury goods will be imposed in 2026."

Labeled Output:

| Token     | Label           |
|-----------|------------------|
| 10        | B-amount         |
| %         | I-amount         |
| tax       | B-policy_tool    |
| luxury    | B-target_group   |
| goods     | I-target_group   |
| 2026      | B-time_period    |

## Technologies

| Step                | Tool                                |
|---------------------|-------------------------------------|
| Data Preparation    | Python + manual labeling (JSON)     |
| Model Architecture  | bert-base-cased                     |
| Training Framework  | Hugging Face Transformers Trainer   |
| Evaluation          | seqeval for token-level F1, precision, recall |
| Visualization       | Confusion matrix (Seaborn + sklearn) |

## Workflow

### 1. Dataset

200 manually labeled sentences are stored in `token_policy_ner_annotated.json`, each with `tokens` and corresponding BIO labels.

### 2. Tokenization & Label Alignment

- Token-level labels are aligned with subword tokens using Hugging Face’s `word_ids()`.
- Subwords are ignored during loss computation (`label = -100`).

### 3. Training Setup

```
TrainingArguments(
    output_dir="./ner-policy",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_steps=10
)
```

### 4. Evaluation Metrics

Metrics are computed at the entity level using `seqeval`:

Precision: 0.83  
Recall:    0.83  
F1-score:  0.83  

These indicate balanced performance — the model is neither underfitting nor overfitting.

## Confusion Matrix

A token-level confusion matrix is used to diagnose misclassifications.

### Observations:
- High accuracy on I-time_period and I-region
- Most confusion between B-amount and B-sector
- Suggests more data is needed for similar-looking entity classes

## What the Model Learns

The model learns to assign a label to each token based on context using a classification head on top of contextual BERT embeddings. Fine-tuning aligns the model’s internal representations with domain-specific tokens (e.g., “subsidy”, “billion”, “2027”).

## Future Work

- Expand training data with diverse sentence structures
- Use domain-adapted language models (e.g., Legal-BERT, PolicyBERT)
- Address overlapping entities with sequence-to-sequence or span-based models

## Files Included

| File                             | Description                          |
|----------------------------------|--------------------------------------|
| token_policy_ner_clean.ipynb     | Full Colab notebook with code        |
| token_policy_ner_annotated.json  | Manually labeled training data       |
| report_token_policy_ner.Rmd      | Training module report (editable)    |
| report_token_policy_ner.pdf      | Final rendered PDF report            |
| README.md                        | This file                            |


