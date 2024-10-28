# Importing Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

# Generate Synthetic Data
np.random.seed(42)
data_size = 1000

# Simulate customer features
data = {
    'tenure': np.random.randint(1, 72, data_size),  # Number of months a customer has stayed
    'monthly_charges': np.round(np.random.uniform(20, 100, data_size), 2),  # Monthly charges
    'total_charges': np.round(np.random.uniform(100, 5000, data_size), 2),  # Total charges
    'contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], data_size),
    'internet_service': np.random.choice(['DSL', 'Fiber optic', 'No'], data_size),
    'payment_method': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], data_size),
    'churn': np.random.choice([0, 1], data_size, p=[0.7, 0.3])  # 0 = no churn, 1 = churn
}

# Create DataFrame
df = pd.DataFrame(data)

# One-hot encoding categorical features
df = pd.get_dummies(df, columns=['contract', 'internet_service', 'payment_method'], drop_first=True)

# Split dataset into features (X) and target variable (y)
X = df.drop('churn', axis=1)
y = df['churn']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Model Training - Logistic Regression
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

# Model Prediction
y_pred = log_reg.predict(X_test)

# Model Evaluation
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)
print("ROC AUC Score:", roc_auc)

# Plotting the ROC Curve
y_pred_prob = log_reg.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'Logistic Regression (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='best')
plt.show()
