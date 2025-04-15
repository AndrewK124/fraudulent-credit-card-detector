# Fraudulent Credit Card Detector

This project is a machine learning-based system for detecting fraudulent credit card transactions using supervised learning techniques. It explores the use of both undersampling and oversampling techniques to address class imbalance, and compares model performance across different sampling techniques. 

-----

## Dataset 

The dataset used is the classic [Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Total Transactions: 284,807
- Fraudulent Transactions: 492
- Features: 30 (28 PCA components, Time, Amount)
- Target: "Class" (0 = Normal, 1 = Fraud)

-----

## My Deeds: 

1. Data Preprocessing:
   - Loaded the dataset and displayed basic statistics.
   - Scaled the "Amount" coulmn used "StandardScaler"
   - Dropped the "Time" column (wasn't relevant to the objective)
   - Checked for and removed duplicate rows.
   - Visualized class imbalance

2. Data Splitting:
   - Separated data into features and target (X and y respectively)
   - Split data into training and testing sets

3. Baseline Modeling:
   - Trained and evaluated:
     1. Logisitic Regression
     2. Decision Tree Classifier
  - Evaluated on original (imbalanced) dataset

4. Undersampling
   - Randomly sampled the majority class to match the minority class count, resulting in massive data loss.
   - Retrained and evaluated models on the balanced dataset.

5. Oversampling (SMOTE)
   - Applied SMOTE to synthetically generate minority class samples.
   - Retrained and evaluated models on the balanced dataset.

6. Model Preservation
- Discovered that a Decision Tree model on the SMOTE data performed the best at detecting fraudulent transactions.
- Saved this model using "joblib"

-----

## Models Used

- Logisitic Regression
- Decision Tree Classifier

Metrics Evaluated: 
- Accuracy
- Precision
- Recall
- F1 Score

-----

## Results Overview 

The models were evaluated across: 
- Original imbalanaced data
- Undersampled balanced data
- SMOTE oversampled balanced data

Performance generally improved in terms of recall and F1 score when data balanced using SMOTE. 

------

## Model Export 
The final model was trained on the SMOTE-balanced dataset.

You can load this model using Python: 
import joblib 
model = joblib.load('credit_card_model.pkl')

------

## Requirements 

pandas 
scikit-learn 
seaborn 
matplotlib
imbalanced-learn 
joblib 

------

## Running the Project 
1. Clone the repository
2. Download the creditcard.csv dataset from Kaggle and place it in the root directory of your project
3. Run the code in either JupyterLab or via VS Code's Jupyter extension

------

I encourage you to research and test more models on this data! 
