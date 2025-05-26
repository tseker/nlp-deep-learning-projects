
# NLP for Historical Policy Text Classification

This repository contains a Jupyter notebook demonstrating the use of natural language processing (NLP) techniques for analyzing historical economic policy texts. The analysis includes both named entity recognition and zero-shot text classification to identify relevant actors, dates, and policy types from trade-related statements.

## Contents

- **Exercise 1:** Named Entity Recognition (NER) with spaCy
- **Exercise 2:** Zero-Shot Classification with HuggingFace Transformers
- **Exercise 3:** NER applied to a list of historical sentences (optional extension)

## Objective

The objective of these exercises is to explore how modern NLP tools can be used to analyze historical narratives involving trade and industrial policy. These tools allow us to:

- Extract structured information (e.g., persons, dates, and policy targets) from raw text
- Classify text based on inferred policy intent (e.g., political, economic, neutral) using a general-purpose transformer model without supervised fine-tuning

## Tools Used

- Python 3.10
- [spaCy](https://spacy.io/)
- [HuggingFace Transformers](https://huggingface.co/transformers/)
- PyTorch (CPU version)

## Setup Instructions

To run this notebook locally, we recommend setting up a virtual environment with the required dependencies:

```bash
# Create and activate a virtual environment
python -m venv nlp-env
source nlp-env/bin/activate  # On Windows use: nlp-env\Scripts\activate

# Install required packages
pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu
pip install transformers spacy
python -m spacy download en_core_web_sm
```

To open the notebook:

```bash
jupyter notebook
```

Then select the `NLP_Policy_Classification_Tuba.ipynb` file.

## License

This work is provided for educational and demonstration purposes and is licensed under the MIT License.
