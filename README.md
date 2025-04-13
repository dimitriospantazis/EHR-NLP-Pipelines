# Medical NLP Insights: Comparative EHR NLP Pipelines

This project demonstrates and compare two Natural Language Processing (NLP) pipelines for Electronic Health Record (EHR) text classification on the MTSamples dataset. The project illustrates how to process clinical transcription data and predict the medical specialty using two approaches:

- **TF-IDF Pipeline:** Uses traditional bag-of-words representations with TF-IDF weighting combined with Logistic Regression.
- **ClinicalBERT Pipeline:** Leverages contextualized embeddings from the Bio_ClinicalBERT model (a version of ClinicalBERT fine-tuned on clinical data) with Logistic Regression.

## Table of Contents

- [Overview](#overview)
- [Pipelines](#pipelines)
  - [TF-IDF Pipeline](#tf-idf-pipeline)
  - [ClinicalBERT Pipeline](#clinicalbert-pipeline)
- [Comparison](#comparison)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Future Work](#future-work)
- [License](#license)

## Overview

The goal of this project is to classify clinical transcriptions into their respective medical specialties. This task is a common application in healthcare informatics, where extracting meaningful insights from unstructured EHR text can improve downstream decision-making and patient care.

We explore two different NLP approaches:

1. **TF-IDF with Logistic Regression:** A classic machine learning method that vectorizes text via Term Frequency-Inverse Document Frequency (TF-IDF) and uses Logistic Regression for classification.
2. **ClinicalBERT with Logistic Regression:** A modern approach using transformer-based contextual embeddings from the Bio_ClinicalBERT model, which captures nuanced clinical language better than traditional methods.

## Pipelines

### TF-IDF Pipeline

- **Data Preprocessing:** The pipeline involves standard text preprocessing techniques like lowercasing, punctuation removal, stopword filtering, and tokenization.
- **Feature Extraction:** TF-IDF vectorization is applied to transform the text data into numerical features.
- **Classification:** A Logistic Regression classifier is used to predict the medical specialty.
- **Evaluation:** The pipeline outputs performance metrics including precision, recall, F1-score, and a confusion matrix.

### ClinicalBERT Pipeline

- **Data Preprocessing:** A minimal cleaning process is applied (trimming whitespace) to prepare the clinical transcriptions.
- **Feature Extraction:** Clinical text is tokenized and embedded using the `emilyalsentzer/Bio_ClinicalBERT` model. Mean pooling is performed over the token embeddings to generate a fixed-length representation.
- **Classification:** The embeddings are then used with a Logistic Regression classifier.
- **Evaluation:** The pipeline computes a classification report to assess its performance.

## Comparison

The project offers a comparative analysis between the traditional TF-IDF approach and the modern transformer-based ClinicalBERT:

- **Interpretability & Speed:**  
  The TF-IDF pipeline is straightforward and computationally efficient for smaller datasets, but it may not capture the complex semantic and contextual relationships inherent in clinical text.

- **Performance & Clinical Relevance:**  
  ClinicalBERT, fine-tuned on biomedical texts, usually provides richer representations and can potentially improve classification performance on complex EHR data. However, it comes with additional computational overhead and latency.

This comparison helps demonstrate the trade-offs between using classical NLP methods and more advanced, deep-learning-based models in healthcare applications.

## Project Structure

```
MedNLP-Insights/
├── clinicalbert_pipeline.ipynb    # Jupyter Notebook for ClinicalBERT pipeline
├── tfidf_pipeline.ipynb           # Jupyter Notebook for TF-IDF pipeline
├── README.md                      # This file
└── requirements.txt               # List of Python dependencies
```


## Installation

1. **Clone this repository:**

   ```bash
   git clone https://github.com/your-username/MedNLP-Insights.git
   cd MedNLP-Insights

2. **Create and activate a virtual environment (optional, but recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

3. **Install the required packages:**

   ```bash
    pip install -r requirements.txt


## Usage
- TF-IDF Pipeline:
    - Open tfidf_pipeline.ipynb in Jupyter Notebook/JupyterLab.
    - Execute each cell sequentially to preprocess the data, perform TF-IDF vectorization, train the classifier, and evaluate the results.

- ClinicalBERT Pipeline:
    - Open clinicalbert_pipeline.ipynb in Jupyter Notebook/JupyterLab.
    - Run the cells to load the model, embed the clinical texts using ClinicalBERT, train the classifier, and examine the performance.

## Results
The notebooks provide classification reports and confusion matrices for both pipelines, which allow you to compare their effectiveness in identifying the correct medical specialty from the clinical text. Evaluating the two approaches highlights the benefits and limitations of traditional NLP methods versus modern deep learning techniques in a clinical context.

## Future Work
Hyperparameter Tuning: Experiment with additional hyperparameters for both Logistic Regression and embedding techniques.

- Model Ensembles: Explore ensemble methods that combine TF-IDF and ClinicalBERT representations.

- Dataset Expansion: Apply the pipelines to larger or more diverse EHR datasets.

- Deep Learning Models: Consider end-to-end deep learning approaches (e.g., fine-tuning ClinicalBERT for classification) for improved performance.

## License
This project is licensed under the MIT License.

