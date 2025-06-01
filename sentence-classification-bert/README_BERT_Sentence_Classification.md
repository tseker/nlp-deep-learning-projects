
# BERT-Based Sentence Classification on Policy Statements

## Overview

This project implements a sentence classification model using BERT to categorize policy-related statements into four classes: `subsidy`, `tax`, `ban`, and `other`. The goal is to apply transformer-based NLP methods to a real-world dataset and evaluate model performance, while documenting the encountered technical challenges and solutions.

## Dataset

The dataset consists of labeled policy sentences (`subsidy`, `tax`, `ban`, `other`). We initially used a small dataset and observed overfitting, so we later expanded it to include 500+ unique and diversified sentences. This significantly improved the generalization performance of the model.

## Objective

The primary objective was to fine-tune a pre-trained BERT model (`bert-base-uncased`) on the classification task. We aimed to achieve reliable performance across all categories, particularly focusing on resolving the model’s overconfidence and class imbalance issues.

## Implementation

The implementation involved the following steps:

1. **Data Loading and Cleaning**
   - Loaded labeled policy sentence data.
   - Removed duplicates and ensured label consistency.

2. **Tokenization**
   - Used `BertTokenizerFast` to tokenize the sentences with truncation and padding.

3. **Model Setup**
   - `BertForSequenceClassification` with 4 output labels.
   - Used the Hugging Face `Trainer` API for training.

4. **Training Parameters**
   - Batch size: 16
   - Learning rate: 2e-5
   - Epochs: 5
   - Optimizer and scheduler automatically handled by Trainer API

5. **Evaluation**
   - Used accuracy and classification report from `sklearn`.
   - Confusion matrix plotted for error analysis.

## Key Results

- The final model achieved high accuracy on the validation set (problem).
- Balanced performance across categories, especially after dataset enhancement.
- Identified early signs of overfitting during experiments with smaller datasets.

## Problems Encountered

###  Package Version Conflicts with Evaluation Strategy

During the training setup, we intended to use parameters like:

```python
evaluation_strategy="epoch"
save_strategy="epoch"
load_best_model_at_end=True
```

However, these resulted in persistent runtime errors due to version incompatibility across `transformers`, `datasets`, and `accelerate`. Specifically:

- **`Trainer` crashed** when evaluation strategies were active without full metric setup.
- **`load_best_model_at_end`** threw errors due to missing validation metric tracking.
- Colab, Jupyter Notebook and VS Code produced inconsistent results due to different dependency versions.

**Temporary Solution:** We commented out the above lines and continued training with a fixed learning rate, omitting validation callbacks.

**Future Recommendation:** Use a clean environment and the following setup:

```bash
pip install transformers==4.38.0 datasets==2.18.0 accelerate==0.26.0
```

## Future Directions

- Implement stratified sampling to improve balance across categories.
- Add early stopping to avoid overfitting.
- Deploy the model as an API for integration with real-world applications.

## File Structure

- `Colab_BERT_Sentence_Classification.ipynb`: Main training and evaluation notebook
- `Labeled_Policy_Sentences.csv`: Cleaned and labeled sentence dataset
- `confusion_matrix.png`: Final confusion matrix visualization

## Conclusion

This project demonstrates how transformer-based models can be fine-tuned for practical sentence classification. Despite technical challenges, we achieved results and created a foundation for more robust future experiments.

