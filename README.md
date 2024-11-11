# Smartphone Price Prediction Model

This repository contains a machine learning model to predict smartphone prices based on various features such as brand, model, specifications, and more. Multiple regression models were tested to find the best-performing one for this task.

## Key Features:
- **Data Preprocessing**: Cleaned and processed data to handle missing values, categorical features, and scaling.
- **Modeling**: Multiple machine learning models were tested and evaluated, including Linear Regression, Ridge, Lasso, Random Forest, and others.
- **Hyperparameter Tuning**: Optimized model performance using Grid Search with cross-validation.
- **Evaluation Metrics**: The model's performance is evaluated using metrics like RMSE, MAE, MSE, and R² to ensure reliable price predictions.

## Technologies Used:
- Python
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Jupyter Notebook

## Objective:
To provide a robust predictive model for smartphone prices that can be utilized in various applications such as e-commerce platforms, price comparison websites, or market analysis tools.

---

## Performance Summary

### Models Used:
- Linear Regression
- Ridge Regression
- Lasso Regression
- ElasticNet Regression
- Support Vector Regression (SVR)
- K-Nearest Neighbors Regression (KNN)
- Random Forest Regression
- Gradient Boosting Regression
- AdaBoost Regression
- Decision Tree Regression
- Multi-layer Perceptron (MLP) Regression

### Performance Evaluation Metrics:
- **R² (Coefficient of Determination)**: A measure of how well the model predicts the target variable. The closer it is to 1, the better the model.
- **RMSE (Root Mean Squared Error)**: A metric that indicates the average magnitude of error. Lower values are better.
- **MAE (Mean Absolute Error)**: The average magnitude of the error in the predictions. Lower values are better.
- **MSE (Mean Squared Error)**: Similar to RMSE, but squares the errors before averaging, emphasizing larger errors.

---

## Model Performance

The following table summarizes the performance of various regression models used in the smartphone price prediction task. The metrics include R² Score, RMSE, MAE, and MSE.

| Model                        | Cross-Validated RMSE      | Mean Absolute Error | Mean Squared Error | R-squared |
|------------------------------|---------------------------|----------------------|---------------------|-----------|
| **Linear Regression**         | 1721841028.1090 ± 3443682055.5570 | 0.1676               | 0.0551              | 0.8804    |
| **Ridge Regression**          | 0.2970 ± 0.0730           | 0.1612               | 0.0560              | 0.8784    |
| **Lasso Regression**          | 0.6142 ± 0.0276           | 0.5527               | 0.4621              | -0.0033   |
| **ElasticNet Regression**     | 0.6142 ± 0.0276           | 0.5527               | 0.4621              | -0.0033   |
| **Support Vector Regression** | 0.2114 ± 0.0183           | 0.1324               | 0.0353              | 0.9234    |
| **K-Nearest Neighbors**       | 0.2771 ± 0.0103           | 0.1902               | 0.0713              | 0.8452    |
| **Random Forest Regression**  | 0.0642 ± 0.0549           | 0.0166               | 0.0163              | 0.9645    |
| **Gradient Boosting Regression**| 0.0464 ± 0.0634          | 0.0165               | 0.0243              | 0.9473    |
| **AdaBoost Regression**       | 0.0726 ± 0.0479           | 0.0511               | 0.0231              | 0.9498    |
| **Decision Tree Regression**  | 0.0553 ± 0.0623           | 0.0165               | 0.0258              | 0.9439    |
| **MLP Regression**            | 0.5961 ± 0.2371           | 0.3008               | 0.1838              | 0.6009    |

---

## Key Insights:
- **Random Forest Regression** stands out with the highest R² score of **0.9645** and the lowest RMSE of **0.0642**, making it the best-performing model for smartphone price prediction.
- **AdaBoost** and **Gradient Boosting** also perform very well with R² values around **0.95**, indicating strong predictive accuracy.
- **Decision Tree** is quite close in performance to **Random Forest**, with an R² of **0.9439**, and a very low MAE of **0.0165**.
- **Support Vector Regression (SVR)** performed well with an R² of **0.9234**, indicating that it generalizes well on unseen data.
- **Lasso** and **ElasticNet** regressions didn't perform as well as the tree-based models, with negative R² indicating a poor fit.

---

## Conclusion:
- **Random Forest** is the model of choice due to its excellent performance in terms of accuracy (**R²**), error metrics (**RMSE**, **MAE**), and stability across cross-validation.
- **AdaBoost** and **Gradient Boosting** are also good alternatives if you want slightly lower error rates and still strong accuracy.
- Tree-based models, such as **Decision Tree** and **Random Forest**, generally provide better performance for this task compared to linear models like **Linear Regression**, **Ridge**, and **Lasso**.
