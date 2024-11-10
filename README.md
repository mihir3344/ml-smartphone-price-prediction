#Smartphone Price Prediction Model
This repository contains a machine learning model for predicting the prices of smartphones based on various features such as brand, model, specifications, and other key attributes. The model is built using regression techniques and is designed to provide an estimate of a smartphone's price based on input data, making it a valuable tool for price forecasting in the mobile phone industry.

Key Features:
Data Preprocessing: Cleaned and processed data to handle missing values, categorical features, and scaling.
Modeling: Multiple machine learning models were tested and evaluated, including Linear Regression, Ridge, Lasso, Random Forest, and others.
Hyperparameter Tuning: Optimized model performance using Grid Search with cross-validation.
Evaluation Metrics: The model's performance is evaluated using metrics like RMSE, MAE, MSE, and R² to ensure reliable price predictions.
Technologies Used:
Python
Scikit-learn
Pandas
NumPy
Matplotlib
Jupyter Notebook
Objective:
To provide a robust predictive model for smartphone prices that can be utilized in various applications such as e-commerce platforms, price comparison websites, or market analysis tools.
# Smartphone Price Prediction Model

This repository contains a machine learning model to predict smartphone prices based on various features such as brand, model, specifications, and more. Multiple regression models were tested to find the best-performing one for this task.

## Performance Summary

### Models Used:
1. **Linear Regression**
2. **Ridge Regression**
3. **Lasso Regression**
4. **ElasticNet Regression**
5. **Support Vector Regression (SVR)**
6. **K-Nearest Neighbors Regression (KNN)**
7. **Random Forest Regression**
8. **Gradient Boosting Regression**
9. **AdaBoost Regression**
10. **Decision Tree Regression**
11. **Multi-layer Perceptron (MLP) Regression**

### Performance Evaluation Metrics:
- **R²** (Coefficient of Determination): A measure of how well the model predicts the target variable. The closer it is to 1, the better the model.
- **RMSE** (Root Mean Squared Error): A metric that indicates the average magnitude of error. Lower values are better.
- **MAE** (Mean Absolute Error): The average magnitude of the error in the predictions. Lower values are better.
- **MSE** (Mean Squared Error): Similar to RMSE, but squares the errors before averaging, emphasizing larger errors.

### Model Performance:
# Model Performance

The following table summarizes the performance of various regression models used in the smartphone price prediction task. The metrics include R² Score, RMSE, MAE, and MSE.

| Model                        | R² Score  | RMSE        | MAE         | MSE         |
|------------------------------|-----------|-------------|-------------|-------------|
| **Linear Regression**         | 0.8718    | 569,726,784.74 ± 1,139,453,568.83 | 0.1706      | 0.0590      |
| **Ridge Regression**          | 0.8793    | 0.2757 ± 0.0357      | 0.1627      | 0.0556      |
| **Lasso Regression**          | -0.0033   | 0.6142 ± 0.0276      | 0.5527      | 0.4621      |
| **ElasticNet Regression**     | -0.0033   | 0.6142 ± 0.0276      | 0.5527      | 0.4621      |
| **Support Vector Regression** | 0.9006    | 0.2478 ± 0.0231      | 0.1524      | 0.0458      |
| **K-Nearest Neighbors**       | 0.8392    | 0.2955 ± 0.0177      | 0.2117      | 0.0740      |
| **Random Forest Regression**  | 0.9738    | 0.0658 ± 0.0559      | 0.0142      | 0.0121      |
| **Gradient Boosting Regression** | 0.9465 | 0.0495 ± 0.0646      | 0.0164      | 0.0246      |
| **AdaBoost Regression**       | 0.9513    | 0.0760 ± 0.0482      | 0.0470      | 0.0224      |
| **Decision Tree Regression**  | 0.9439    | 0.0501 ± 0.0625      | 0.0163      | 0.0258      |
| **MLP Regression**            | 0.8003    | 0.3166 ± 0.0214      | 0.2082      | 0.0920      |



### Key Insights:
- **Random Forest Regression** stands out with the highest **R² score** of **0.9712** and the lowest **RMSE** of **0.0658**, making it the best performing model for smartphone price prediction.
- **AdaBoost** and **Gradient Boosting** also perform very well with **R²** values around **0.95**, indicating strong predictive accuracy.
- **Decision Tree** is quite close in performance to **Random Forest**, with a **R²** of **0.9439**, and a very low **MAE** of **0.0162**.
- **Support Vector Regression (SVR)** performed well with a **R²** of **0.9041**, indicating that it generalizes well on unseen data.
- **Lasso** and **ElasticNet** regressions didn't perform as well as the tree-based models, with **negative R²** indicating a poor fit.

### Conclusion:
- **Random Forest** is the model of choice due to its excellent performance in terms of **accuracy (R²)**, **error metrics (RMSE, MAE)**, and stability across cross-validation.
- **AdaBoost** and **Gradient Boosting** are also good alternatives if you want slightly lower error rates and still strong accuracy.
- Tree-based models, such as **Decision Tree** and **Random Forest**, generally provide better performance for this task compared to linear models like **Linear Regression**, **Ridge**, and **Lasso**.

