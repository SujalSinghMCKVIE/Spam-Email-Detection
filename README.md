# Spam Email Filtering: Ham or Spam?

A machine learning-based spam email classifier that detects and filters unwanted emails. It follows a structured approach, including data cleaning, exploratory data analysis (EDA), text preprocessing, model building, and evaluation using various classifiers such as Naïve Bayes, SVM, and XGBoost. The goal is to enhance email security by leveraging NLP and ML techniques.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Models Used](#models-used)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Future Improvements](#future-improvements)
- [License](#license)

## Introduction
Spam emails are a major issue in modern communication, causing security risks and unnecessary clutter. This project aims to build an efficient spam email filtering system using machine learning algorithms to classify emails as spam or ham (not spam).

## Dataset
- The dataset contains **5,572 emails**, categorized as spam or ham.
- Columns: `Category` (target variable: ham/spam), `Message` (email content).
- Preprocessing steps include cleaning text, tokenization, stemming, and vectorization.

## Project Workflow
1. **Data Checks** – Checking for missing values, duplicates, and inconsistencies.
2. **Data Cleaning** – Removing duplicates, formatting text, and handling null values.
3. **EDA (Exploratory Data Analysis)** – Understanding email distributions, word counts, and patterns.
4. **Data Preprocessing** – Tokenization, stemming, and stopword removal.
5. **Model Building** – Training and testing multiple machine learning models.
6. **Model Evaluation** – Comparing accuracy, precision, recall, and F1-score.

## Technologies Used
- **Programming Language:** Python
- **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, NLTK, WordCloud, XGBoost

## Installation
To set up the project, follow these steps:

```bash
# Clone the repository
git clone https://github.com/yourusername/spam-email-filtering.git
cd spam-email-filtering

# Install dependencies
pip install -r requirements.txt
```

## Usage
Run the main script to train and evaluate the model:
```bash
python spam_filter.py
```

## Models Used
- Naïve Bayes
- Support Vector Machine (SVM)
- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

## Evaluation Metrics
- **Accuracy**: Measures overall correctness.
- **Precision**: Measures correctness of spam predictions.
- **Recall**: Measures ability to detect actual spam.
- **F1-Score**: Balances precision and recall.

## Results
The **XGBoost** and **Gradient Boosting Decision Trees (GBDT)** models performed the best, achieving high accuracy and balanced precision-recall scores.

## Future Improvements
- Implement deep learning models (LSTMs, Transformers).
- Improve text preprocessing with more advanced NLP techniques.
- Train on a larger dataset for better generalization.

## License
This project is licensed under the MIT License. See `LICENSE` for details.
