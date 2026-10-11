# E-Commerce Review Analyzer and Product Rating Predictor

Student: Sujan KS (1AY25MC065)
Guide: Prof. Sumit Singha Chowdhury
Institution: Acharya Institute of Technology, Department of MCA
Semester: 2nd and 3rd

## Overview

A multi-task deep learning system that analyzes e-commerce product reviews to:
1. Classify sentiment (positive, negative, neutral)
2. Predict star ratings (1–5)
3. Extract product aspects
4. Provide explainable predictions

## Architecture

- Baseline: TF-IDF + Logistic Regression / Linear SVM
- Main model: Shared RoBERTa encoder + two task-specific heads (multi-task learning)
- Dashboard: Streamlit interface with visualizations and explanations

## Dataset

- Source: Amazon Reviews 2023 (McAuley Lab, UC San Diego)
- Categories: Electronics, Cell Phones and Accessories, Clothing Shoes and Jewelry
- Size: 60,000 balanced reviews (4,000 per rating class per category)
- Split: 70/15/15 stratified

## Progress

- Phase 1: Dataset pipeline (download, clean, balance, split, EDA)
- Phase 2: Baseline models (TF-IDF + LR/SVM)
- Phase 3: RoBERTa multi-task model (in progress)
- Phase 4: Aspect extraction + explainability
- Phase 5: Streamlit dashboard
- Phase 6: Final report

## Baseline Results

| Model | Task | Accuracy | Macro-F1 | MAE |
|---|---|---|---|---|
| LR + TF-IDF | Sentiment | 0.66 | 0.617 | — |
| SVM + TF-IDF | Sentiment | 0.66 | 0.599 | — |
| LR + TF-IDF | Rating | 0.47 | 0.47 | 0.761 |
| SVM + TF-IDF | Rating | 0.44 | 0.43 | 0.831 |

Key finding: Neutral class F1 = 0.38 — this is the main challenge the transformer model will address.

## Tools

- Python, PyTorch, Hugging Face Transformers
- scikit-learn (baseline), NLTK
- Streamlit (dashboard)
- Google Colab (development)

## Repository Structure
├── synopsis/ # Project synopsis document
├── notebooks/ # Colab notebooks (Phase 1, Phase 2)
├── results/ # Charts, metrics, confusion matrices
├── data/ # Sample data (full data on Google Drive)
└── README.md


## Contact

- Student: sujusuresh2022@gmail.com
- Guide: sumit2432@acharya.ac.in
