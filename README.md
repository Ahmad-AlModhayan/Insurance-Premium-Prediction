# Insurance Premium Prediction

This project implements a custom Linear Regression model to predict insurance premiums based on various customer features. The implementation includes L2 regularization for better generalization and uses gradient descent for optimization.

## Mathematical Foundation

### Linear Regression Model
The model predicts insurance charges using the following equation:
y = Xw + b
where:
- y is the predicted insurance premium
- X is the feature matrix
- w is the weight vector
- b is the bias term

### Cost Function
We use Mean Squared Error (MSE) with L2 regularization:
J(w,b) = (1/2m) * [Σ(y_pred - y)² + λΣw²]
where:
- m is the number of samples
- λ (lambda_reg) is the regularization parameter
- The first term is the MSE
- The second term is the L2 regularization

### Gradient Descent
Parameters are updated using:
w = w - α * ∂J/∂w
b = b - α * ∂J/∂b
where:
- α (learning_rate) is the step size
- ∂J/∂w and ∂J/∂b are the gradients

## Setup Instructions

1. Clone the repository
2. Ensure you have the required dependencies:
   ```
   numpy
   pandas
   matplotlib
   seaborn
   scikit-learn
   ```
3. Place the data files in the `data/` directory:
   - train.csv
   - test.csv
   you can get the data from [Kaggle](https://www.kaggle.com/competitions/playground-series-s4e12/data)

## Usage

Run the notebook `insurance_premium_prediction.ipynb`


The script will:
1. Load and preprocess the data
2. Train the custom linear regression model
3. Compare performance with scikit-learn's implementation
4. Generate visualization plots:
   - cost_history.png: Shows the convergence of the cost function
   - predictions.png: Displays actual vs predicted premiums

## Model Features

- **L2 Regularization**: Prevents overfitting by penalizing large weights
- **Gradient Descent**: Optimizes model parameters iteratively
- **Feature Scaling**: Standardizes features for better convergence
- **Categorical Encoding**: Uses one-hot encoding for categorical variables

## Performance Metrics

The model's performance is evaluated using:
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

Results are compared with scikit-learn's implementation for validation.

## Data Processing

1. **Feature Engineering**:
   - One-hot encoding for categorical variables (sex, smoker, region)
   - Standardization of numerical features

2. **Data Split**:
   - 80% training data
   - 20% validation data

## Hyperparameters

- Learning rate: 0.01
- Number of iterations: 1000
- L2 regularization parameter: 0.1

These parameters can be adjusted in the LinearRegression class initialization.
