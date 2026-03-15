import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
# 1. read the data from the file
def read_California_dataset() -> pd.DataFrame:
    # df-->DataFrame, 
    # pd.read_csv()-->function to read a CSV file and create a DataFrame from it.
    df = pd.read_csv("California_Houses.csv")
    return df
# 2. split the dataset into training, validation and testing sets with stratification
def split_data(df: pd.DataFrame, training_fractor: float = 0.7, validation_fractor: float = 0.15, testing_fractor: float = 0.15,):
    # checking if the sum of the fractions is equal to 1, otherwise it will raise an error
    assert abs(training_fractor + validation_fractor + testing_fractor - 1.0) < 1e-6
    # X-->features, y-->labels
    # .values-->returns the values of the DataFrame as a NumPy array
    # .drop()-->removes the class column from the DataFrame
    X = df.drop(columns=["Median_House_Value"])
    # df["class"] == "g"-->returns a boolean array where True if the class is 'g' and False otherwise
    # .astype(int)-->converts boolean values to integers (True-->1, False-->0)``
    y = (df["Median_House_Value"])
    np.random.seed(42)
    # train_test_split-->built-in function that shuffle the data and split the sets(X, Y) into 4 sets(X1, X2, y1, y2)
        # and ensuring that with every split, the same rows are selected for the same set 
        # example-->if row 40 in X is selected for the training set, it will also be selected for the training set in Y
    X_training, X_temp, y_training, y_temp = train_test_split(X, y, train_size=training_fractor)
    # split the remaining data evenly into validation and test sets
    X_validation, X_testing, y_validation, y_testing = train_test_split(X_temp, y_temp, train_size=validation_fractor / (validation_fractor + testing_fractor))
    return X_training, X_validation, X_testing, y_training, y_validation, y_testing
# 3. train and evaluate the models
def train_and_evaluate_models(X_training, y_training, X_validation, y_validation, X_testing, y_testing):
    # It is a best practice to scale the data for regression
    # StandardScaler-->standardizes features by removing the mean and scaling to unit variance
    # after scaling every column will have a mean of 0 and a standard deviation of 1
    # helps the model to converge faster and prevents overfitting
    # forces the model to pay equal attention to all features
    #  as if it is normalized 
    # important for regularized models like Lasso and Ridge so penalties are applied equally
    scaler = StandardScaler()
    #  fit --> when scalar sees X_training, it calculates the mean and standard deviation of each feature(column)
            #  they are saved in the scaler object's memory
    # transform()--> apply the formula (x - mean) / std to each value in the data to scale it
    # we use .fit_tamsform on training set--> to learn the parameters from the training data
    # we use .transform on validation and testing set--> to apply the same transformation to the validation and testing data
                #  we can't use .fit cause we don't want the model to learn from the validation and testing data
                #  cause they are supposed to be unseen data
    X_training_scaled = scaler.fit_transform(X_training)
    # by learning the parameters (rules) from the training data, we can apply the same transformation to the validation and testing data
    #  this ensures that the validation and testing data are scaled in the same way as the training data
    X_validation_scaled = scaler.transform(X_validation)
    X_testing_scaled = scaler.transform(X_testing)

    print("--- Tuning Models on Validation Set ---")
    
    # 3a. Linear Regression 
    #  linear regression is a statistical method that allows us to find the best fit line for a set of data
    # creates an empty linear regression model
    #  tells Scikit-learn to use the Linear Regression algorithm
    lr = LinearRegression()
    # fits the model to the training data
    #  the model learns the relationship between the features and the target variable
    #  it calculates the coefficients and the intercept that minimize the (loss) mean squared error
    lr.fit(X_training_scaled, y_training)
    
    # 3b. Tuning Lasso & Ridge Regression
    # We will try a range of alpha (regularization strength) values
    alphas = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    
    best_lasso = None
    # float('inf')-->infinity --> cause  we want to find the minimum 
    best_lasso_mean_squared_error = float('inf')
    
    best_ridge = None
    best_ridge_mean_squared_error = float('inf')

    # Find best Lasso (L1 regularization) model on Validation set
    # lasso tries to shrink the coefficients of the unimportant features to zero
    # this helps in feature selection
    # penalizes the model by adding the absolute sum of the coefficients to the error equation.
    # lasso's equation --> MSE + alpha * sum(|w_i|)
    for alpha in alphas: 
        # alpha: hyperparameter that controls the strength of regularization (penalty)
        # lasso-->uses iterative algorithm that walk slowly toward the best fit line (solution) step by step
        # max_iter: maximum number of iterations to train the model
        lasso = Lasso(alpha=alpha, max_iter=10000)
        # the training step
        # run the iterative algo to find the best coef weights for each feature
        # after this line the lasso object has the best coef weights for the current alpha and knows how to predict house prices
                    #  based on the pattern it learned from the training data
        lasso.fit(X_training_scaled, y_training)
        # predict the target variable for the validation set
        # for checking the performance of the model on the validation set
        predictions = lasso.predict(X_validation_scaled)
        # calculate the mean squared error
        # formula --> (1/n) * sum((y_true - y_pred)^2)
        mean_squared_errors = mean_squared_error(y_validation, predictions)
        # if the current mean_squared_error is less than the best mean_squared_error found so far
        # update the best mean_squared_error and the best model
        if mean_squared_errors < best_lasso_mean_squared_error:
            best_lasso_mean_squared_error = mean_squared_errors
            best_lasso = lasso

    # Find best Ridge (L2 regularization) model on Validation set
    # penalizes the model by adding the squared sum of the coefficients to the error equation.
    # smooth fyunction that have a convex shape --> a single minimum --> use calculus to instantly find the minimum
    # ridge's equation --> MSE + alpha * sum(w_i^2)
    for alpha in alphas:
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_training_scaled, y_training)
        predictions = ridge.predict(X_validation_scaled)
        mean_squared_errors = mean_squared_error(y_validation, predictions)
        if mean_squared_errors < best_ridge_mean_squared_error:
            best_ridge_mean_squared_error = mean_squared_errors
            best_ridge = ridge
    # print the best alpha for lasso and ridge
    print(f"Best Lasso alpha found: {best_lasso.alpha}")
    print(f"Best Ridge alpha found: {best_ridge.alpha}\n")

    # 4. Report Mean Squared Error and Mean Absolute Errors for all models on Test Set
    print("--- Evaluation on Test Set ---")
    
    models = {
        "Linear Regression": lr,
        f"Lasso Regression (alpha={best_lasso.alpha})": best_lasso,
        f"Ridge Regression (alpha={best_ridge.alpha})": best_ridge
    }

    for name, model in models.items():
        # pedict the output of the scaled testing set
        # how do we predict --> y = w1*x1 + w2*x2 + ... + wn*xn + b
            #  we already know the weights from the training set
            #  we plug in the values of the testing set into the equation
            #  and the model will calculate the predicted values
        y_testing_prediction = model.predict(X_testing_scaled)
        # calculate the mean squared error
        # formula --> (1/n) * sum((y_true - y_pred)^2)
        mean_squared_errors = mean_squared_error(y_testing, y_testing_prediction)
        # calculate the mean absolute error
        # formula --> (1/n) * sum(|y_true - y_pred|)
        mean_absolute_errors = mean_absolute_error(y_testing, y_testing_prediction)
        
        print(f"{name}:")
        print(f"  Mean Squared Error (mean_squared_error) : {mean_squared_errors:,.2f}")
        print(f"  Mean Absolute Error (MAE): {mean_absolute_errors:,.2f}")
        print()

    # 4. Comments on the results and compare between models
    print("--- Final Analysis & Model Comparison ---")
    print("""
    1. Linear Regression: Acts as our baseline. It models the direct linear relationships
       between the features (like median income, rooms) and the house value.
    
    2. Lasso Regression: Adds L1 regularization, which penalizes the absolute size of the coefficients.
       This can force some feature weights to exactly zero, effectively performing feature selection
       and creating a simpler, more interpretable model if some features are irrelevant.
       
    3. Ridge Regression: Adds L2 regularization, which penalizes the squared size of the coefficients.
       This shrinks coefficients evenly and prevents any single feature from dominating the model,
       which is very helpful when there is multicollinearity (highly correlated features).
       
    Comparison Results:
    Depending on the outputted metrics, you'll generally notice:
    - If all models perform extremely similarly, it indicates that the standard linear model wasn't 
      overfitting heavily to begin with.
    - Ridge often slightly outperforms pure Linear Regression in real-world dense datasets by reducing variance.
    - Notice that mean_squared_error is mathematically much larger than mean_absolute_error. This is because mean_squared_error squares the errors 
      before averaging them, meaning mean_squared_error heavily punishes large outlier mistakes (e.g., drastically mispricing).
    """)

def main():
    df = read_California_dataset()
    X_training, X_validation, X_testing, y_training, y_validation, y_testing = split_data(df)
    train_and_evaluate_models(X_training, y_training, X_validation, y_validation, X_testing, y_testing)

if __name__ == "__main__":
    main()
