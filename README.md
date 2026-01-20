# Mental Health Text Classification via DistilBERT & ML Pipelines

## Project Overview
This project focuses on the automated classification of mental health-related text into categories such as "SuicideWatch" and "Depression." The goal was to build a robust NLP pipeline that outperforms traditional machine learning baselines while prioritizing Responsible AI through rigorous error analysis.

## Key Results
* **Performance Gain:** Improved classification accuracy from **82% to 86%** by transitioning from baseline models to a transformer-based architecture.
* **Optimization:** Achieved a **0.86 Macro-F1 score**, significantly enhancing the model's ability to detect subtle crisis indicators and implicit suicidal ideation.

## Technical Implementation
* **Models:** Evaluated Logistic Regression, SVM, SGD, and **DistilBERT**.
* **Engineering:** Developed a custom `OnTheFlyDataset` (PyTorch `Dataset` wrapper) to handle on-the-fly tokenization, avoiding memory bottlenecks during the training of large-scale datasets.
* **Preprocessing:** Implemented full pipelines for data cleaning, tokenization, and **TF-IDF** feature extraction.

## Responsible AI & Error Analysis
A core component of this project was investigating "why" models fail. I conducted a deep-dive error analysis to identify:
* **Label Noise:** Ambiguities in datasets like the "teenagers" subreddit where serious mental health discussions overlap with general venting.
* **Bias Detection:** Identifying misclassifications caused by figurative language and slang.

