Customer Churn Prediction
This project predicts whether a customer is likely to churn (leave the service) based on historical data, using Logistic Regression. Predicting churn helps companies take preventive actions to retain their customers and reduce revenue loss. The current model provides insights but requires further tuning and adjustments due to class imbalance issues.

Table of Contents
Project Overview
Dataset
Project Structure
Installation
Usage
Model Evaluation
Results
Future Improvements
Contributing
License

Project Overview
Customer churn prediction is essential for businesses to understand why customers leave and to implement strategies to retain them. This project uses logistic regression, a classification model suited for binary outcomes, to predict customer churn. The model has been tested on synthetic data, but initial evaluations suggest opportunities for improvement, particularly due to class imbalance.

Dataset
For this project, a synthetic dataset is generated directly within the code to simulate customer information, including:

Tenure: Duration of the customer’s association in months
Monthly Charges: Monthly expenses incurred by the customer
Total Charges: Total amount paid over the entire tenure
Contract Type: Monthly, One year, or Two-year contracts
Internet Service: Type of internet service (DSL, Fiber optic, No service)
Payment Method: Method of payment (Electronic check, Mailed check, Bank transfer, Credit card)
Churn: Target variable indicating if the customer churned (1) or stayed (0)
Project Structure
bash
Copy code
├── churn_prediction.ipynb   # Main Jupyter Notebook
├── README.md                # Project documentation
└── requirements.txt         # Dependencies
Installation
Clone the repository:

bash
Copy code
git clone https://github.com/yourusername/customer-churn-prediction.git
cd customer-churn-prediction
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Usage
Open the churn_prediction.ipynb file in Jupyter Notebook to run the project.
Run the cells in sequence to:
Generate synthetic data
Preprocess the data
Train a logistic regression model
Evaluate the model’s performance
View the evaluation metrics, including accuracy, confusion matrix, and ROC-AUC score, and visualize the ROC curve.
Model Evaluation
The model’s performance is evaluated on several metrics:

Accuracy: The model achieves an accuracy of 71.67%. However, due to class imbalance, accuracy alone does not provide a complete performance picture.
Confusion Matrix: Shows that the model correctly identifies all non-churners but misses all churn cases.
Classification Report: Highlights 0 precision and recall for churners, indicating a need for further optimization.
ROC Curve: Displays the trade-off between the true positive and false positive rates, with an ROC-AUC score of 0.5, suggesting performance comparable to random guessing.
Results
The logistic regression model currently:

Correctly identifies non-churn cases but fails to predict churn cases due to class imbalance.
Requires class balancing, feature engineering, or alternative model testing for more reliable churn prediction.
Future Improvements
Address Class Imbalance:

Implement oversampling (e.g., SMOTE) or undersampling techniques.
Adjust class weights in the logistic regression model.
Feature Engineering:

Add or derive new features, such as average monthly charges or recent interactions, to increase predictive power.
Model Selection:

Experiment with algorithms better suited to imbalanced datasets, such as Random Forest or Gradient Boosting, and compare results.
Hyperparameter Tuning:

Tune logistic regression hyperparameters, such as regularization strength, to prevent overfitting on non-churn cases.
Contributing
Contributions are welcome! If you’d like to improve this project, feel free to fork the repository, make changes, and submit a pull request.

License
This project is open-source and available under the MIT License.


