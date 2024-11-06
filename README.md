# ProductReviewSentiment

# Sentiment Analysis on Product Reviews

This project aims to analyze and classify sentiments from product reviews using various machine learning models, including Naive Bayes, Logistic Regression, and Support Vector Machines (SVM), with feature engineering techniques like Bag-of-Words (BoW) and TF-IDF.

## Table of Contents
- [Dataset](#dataset)
- [Project Steps](#project-steps)
  - [1. Data Loading and Exploration](#1-data-loading-and-exploration)
  - [2. Data Cleaning](#2-data-cleaning)
  - [3. Text Preprocessing](#3-text-preprocessing)
  - [4. Feature Engineering](#4-feature-engineering)
  - [5. Model Training and Evaluation](#5-model-training-and-evaluation)
- [Results](#results)
- [Usage](#usage)
- [License](#license)

## Dataset
The dataset consists of customer product reviews with columns:
- `Review`: Text of the product review.
- `product_price`: Price of the product.
- `Rate`: Rating provided by the customer.
- `Sentiment`: Sentiment label (0, 1, or 2).

## Project Steps

### 1. Data Loading and Exploration
- Mounted Google Drive to access the dataset.
- Extracted and loaded the dataset into a Pandas DataFrame.
- Displayed the first few rows and the structure of the dataset for initial exploration.

### 2. Data Cleaning
- Handled non-numeric values in `product_price` by coercing them to NaN and then dropping rows with NaN values.
- Converted `Rate` to an integer type.
- Removed rows with null values in the `Review` column and filled missing values in the `Summary` column.

### 3. Text Preprocessing
- Cleaned the `Review` text by removing punctuation and special characters.
- Converted text to lowercase and tokenized it into individual words.
- Removed stop words and applied lemmatization for reducing words to their base forms.
- Reconstructed the cleaned text for each review.

### 4. Feature Engineering
- Used `CountVectorizer` for Bag-of-Words (BoW) representation of the text data.
- Employed `TfidfVectorizer` for TF-IDF feature extraction.
- Encoded the `Sentiment` labels using `LabelEncoder`.

### 5. Model Training and Evaluation
- **Naive Bayes**:
  - Trained a Multinomial Naive Bayes model and evaluated it on the test set.
- **Logistic Regression**:
  - Trained a Logistic Regression model and generated a classification report.
- **SVM with Grid Search**:
  - Conducted hyperparameter tuning with GridSearchCV to find the best parameters for SVM.
  - Retrained SVM with optimal parameters and evaluated the model performance.

## Results

### Naive Bayes
- **Accuracy**: 0.90
- **Classification Report**:
  | Class         | Precision | Recall | F1-Score | Support |
|---------------|-----------|--------|----------|---------|
| 0             | 0.85      | 0.71   | 0.77     | 4894    |
| 1             | 0.06      | 0.00   | 0.01     | 1763    |
| 2             | 0.91      | 0.99   | 0.95     | 29420   |
| **Accuracy**  |           |        | 0.90     | 36077   |
| **Macro Avg** | 0.61      | 0.57   | 0.58     | 36077   |
| **Weighted Avg** | 0.86   | 0.90   | 0.88     | 36077   |


### Logistic Regression
- **Accuracy**: 0.91
- **Classification Report**:
- | Class         | Precision | Recall | F1-Score | Support |
|---------------|-----------|--------|----------|---------|
| 0             | 0.85      | 0.75   | 0.80     | 4894    |
| 1             | 0.00      | 0.00   | 0.00     | 1763    |
| 2             | 0.92      | 0.99   | 0.95     | 29420   |
| **Accuracy**  |           |        | 0.91     | 36077   |
| **Macro Avg** | 0.59      | 0.58   | 0.58     | 36077   |
| **Weighted Avg** | 0.86   | 0.91   | 0.88     | 36077   |



### SVM
- **Best Parameters**: `{'C': 1, 'kernel': 'linear'}`
- **Accuracy**: 0.91
- **Classification Report**:
| Metric      | Precision | Recall | F1-Score | Support |
|------------------|-----------|--------|----------|---------|
| Class 0          | 0.85      | 0.75   | 0.80     | 4894    |
| Class 1          | 0.00      | 0.00   | 0.00     | 1763    |
| Class 2          | 0.92      | 0.99   | 0.95     | 29420   |
| **Accuracy**     |           |        | 0.91     | 36077   |
| **Macro Avg**    | 0.59      | 0.58   | 0.58     | 36077   |
| **Weighted Avg** | 0.86      | 0.91   | 0.88     | 36077   |


## Usage
To replicate this project:
1. Load your data into a compatible environment like Google Colab.
2. Follow the code and steps in the project to preprocess, train, and evaluate your models.
3. Save the best model (`best_svm_sentiment_model.pkl`) using joblib for deployment.

## License
This project is licensed under the MIT License.
