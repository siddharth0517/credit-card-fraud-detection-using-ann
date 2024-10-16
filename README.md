# Credit Card Fraud Detection using Artificial Neural Networks (ANN)
This project aims to detect fraudulent transactions in banking using artificial neural networks (ANN). The dataset used is a highly imbalanced credit card transaction dataset, which makes fraud detection more challenging but crucial in real-world applications.

## Project Overview
Fraudulent transactions are a major concern in the banking industry. This project utilizes deep learning techniques, specifically an ANN model, to classify transactions as either fraudulent or legitimate. The model is built, trained, and evaluated to ensure its effectiveness in detecting rare fraud cases.

### Key Highlights:
+ Dataset: Credit card fraud detection dataset from [Kaggle.](https://www.kaggle.com/datasets/gungunshukla15/credit-card-fraud-detection)
+ Model: Artificial Neural Network (ANN) with multiple hidden layers.
+ Goal: Classify transactions into fraudulent or legitimate categories.
+ Techniques: Imbalanced data handling, feature scaling, binary classification.
+ Libraries: TensorFlow/Keras, Scikit-learn, Pandas, NumPy.

## Dataset
The dataset consists of transactions made by European cardholders in September 2013. It contains:

**492 fraudulent transactions** out of 284,807 total transactions, making it highly imbalanced.

## Features:
+ Time: Time elapsed between the first transaction and each transaction.
+ V1-V28: PCA-transformed features to anonymize sensitive information.
+ Amount: Transaction amount.
+ Class: Target variable (0 = Legitimate, 1 = Fraudulent).

## Project Workflow
### 1. Data Preprocessing
+ Handling Imbalanced Data: Used SMOTE (Synthetic Minority Over-sampling Technique) to handle the imbalanced dataset.
+ Feature Scaling: Standardized Amount and Time using StandardScaler from Scikit-learn.
+ Train/Test Split: Data was split into training (80%) and testing (20%) sets.

### 2. Model Building
+ Architecture: Built a fully connected ANN with:
+ Input layer matching the number of features.
+ 2 Hidden layers with ReLU activation.
+ Output layer with Sigmoid activation for binary classification.
+ Optimizer: adam, with binary cross-entropy as the loss function.

### 3. Training the Model
The model was trained using the processed data for 50 epochs with a batch size of 32, along with early stopping to prevent overfitting.

### 4. Model Evaluation
+ Confusion Matrix: Used to evaluate model performance.
+ Accuracy, Precision, Recall, and F1-Score: Metrics were calculated to assess the model’s ability to detect fraudulent transactions.

### Results
+ The model was able to effectively classify fraudulent transactions with reasonable accuracy.
+ F1-Score and Recall were emphasized due to the imbalanced nature of the dataset, where identifying fraudulent cases is critical.

### Future Work
+ Visualization: Future steps can include visualizing model performance through ROC curves, precision-recall curves, and confusion matrices.
+ Model Deployment: The model can be deployed as a web application using Streamlit or Flask, allowing users to input transaction data and receive real-time fraud detection results.
+ Hyperparameter Tuning: Further hyperparameter optimization can be conducted to improve model performance.

Author
[SIDDHARTH JAISWAL]
