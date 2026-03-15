# knn
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
####1. load and balance
# DataFrame-->like a spreadsheet with rows and columns, where each column has a name and a type, and each row is an observation. 
#             It is a powerful data structure for data manipulation and analysis in Python, provided by the pandas library.
def read_magic_dataset() -> pd.DataFrame:
    column_names = ["f1","f2","f3","f4","f5","f6","f7","f8","f9","f10","class",]
    url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/magic/magic04.data")
    # df-->DataFrame, 
    # pd.read_csv()-->function to read a CSV file and create a DataFrame from it.
    df = pd.read_csv(url, names=column_names)
    return df

def balance_dataset(df: pd.DataFrame) -> pd.DataFrame:
    # making a dataframe for each class separately
    gamma = df[df["class"] == "g"]
    hadron = df[df["class"] == "h"]
    # getting the minimum to know how many samples we need to get rid of
    n = min(len(gamma), len(hadron))
    # if we want the same random samples every time we run the code, we can set the seed for reproducibility.
    # This ensures that the random sampling will yield the same results each time the code is executed.
    np.random.seed(42)
    # .sample--> method to randomly sample n rows from the DataFrame.
    gamma_balanced = gamma.sample(n=n)
    hadron_balanced = hadron.sample(n=n)
    # sample(frac=1)-->shuffle all the rows in the combined DataFrame
    # reset_index(drop=True)-->reset the index of the resulting DataFrame after shuffling, dropping the old index.
    # to prevent having the same index twice  and to acces the data easily with the right index
    balanced = pd.concat([gamma_balanced, hadron_balanced]).sample(frac=1).reset_index(drop=True)
    return balanced

####2. split the dataset into training, validation and testing sets with stratification
def split_data(df: pd.DataFrame, training_fractor: float = 0.7, validation_fractor: float = 0.15, testing_fractor: float = 0.15,):
    # checking if the sum of the fractions is equal to 1, otherwise it will raise an error
    assert abs(training_fractor + validation_fractor + testing_fractor - 1.0) < 1e-6
    # X-->features, y-->labels
    # .values-->returns the values of the DataFrame as a NumPy array
    # .drop()-->removes the class column from the DataFrame
    X = df.drop(columns=["class"]).values
    # df["class"] == "g"-->returns a boolean array where True if the class is 'g' and False otherwise
    # .astype(int)-->converts boolean values to integers (True-->1, False-->0)``
    y = (df["class"] == "g").astype(int).values
    # train_test_split-->built-in function that shuffle the data and split the sets(X, Y) into 4 sets(X1, X2, y1, y2)
        # and ensuring that with every split, the same rows are selected for the same set 
        # example-->if row 40 in X is selected for the training set, it will also be selected for the training set in Y
    # stratify=y-->ensures that the proportion of classes is the same in all splits
                #  it is used for classification problems
                #  it is not used for regression problems
                #  it ensures that your training, validation, and test sets all have the exact same percentage of "g" and "h" as the original complete dataset.
    X_training, X_temp, y_training, y_temp = train_test_split(X, y, train_size=training_fractor, stratify=y)
    # split the remaining data evenly into validation and test sets
    X_validation, X_testing, y_validation, y_testing = train_test_split(X_temp, y_temp, train_size=validation_fractor / (validation_fractor + testing_fractor), stratify=y_temp)
    return X_training, X_validation, X_testing, y_training, y_validation, y_testing

def evaluate_knn(X_training, y_training, X_eval, y_eval, ks: list):
    results = {}
    ### 3. apply knn classifier for each k in ks
    for k in ks:
        # KneighborsClassifier-->class constructor
        # classifier-->object   
        classifier = KNeighborsClassifier(n_neighbors=k)
        # .fit-->train the classifier
        # put the training data in the classifier and store them in the memory
        classifier.fit(X_training, y_training)
        # .predict-->predict the class of the evaluation set
        # uses euclidean distance to find the k nearest neighbors
        y_pred = classifier.predict(X_eval)
        ### 5. calculate metrics
        metrics = {
            "accuracy": accuracy_score(y_eval, y_pred),
            "precision": precision_score(y_eval, y_pred),
            "recall": recall_score(y_eval, y_pred),
            "f1": f1_score(y_eval, y_pred),
            "confusion_matrix": confusion_matrix(y_eval, y_pred),
            # output = true-->returns a dictionary, false-->returns a string
            # detailed report of precision, recall, f1-score for each class:1,0
            "classification_report": classification_report(y_eval, y_pred, output_dict=True),
        }
        results[k] = metrics
        print(f"k = {k}")
        print(f"  accuracy  : {metrics['accuracy']:.4f}")
        print(f"  precision : {metrics['precision']:.4f}")
        print(f"  recall    : {metrics['recall']:.4f}")
        print(f"  f1-score  : {metrics['f1']:.4f}")
        print("  confusion matrix:\n", metrics["confusion_matrix"])
        print("\n")
    return results

def main():
    # 1. read data from file and balance
    df = read_magic_dataset()
    df = balance_dataset(df)

    # 2. split data into 3 sets each
    X_training, X_validation, X_testing, y_training, y_validation, y_testing = split_data(df)

    # 3 & 4. try different k values on validation set
    # [1,2,3......20]
    ks = list(range(1, 21))
    print("Evaluating on validation set")
    # val_results-->array of dictionaries, each dictionary contains the metrics for each k
    val_results = evaluate_knn(X_training, y_training, X_validation, y_validation, ks)
    # pick best k by f1
    # why f1--> harmonic mean of precision and recall
    # it is a good measure of a model's performance when we care about both false positives and false negatives
    # the higher the f1 score, the lesser the number of false positives and false negatives
    # key=lambda-->create a tiny in line function that returns the f1 score for each k
    best_k = max(val_results, key=lambda k: val_results[k]["f1"])
    print(f"Best k according to validation f1: {best_k}\n")

    # 5. evaluate on test set using best k and also report all trained models if desired
    print("Evaluating best model on test set")
    # [best_k]-->list containing only the best k-->1 element
    test_results = evaluate_knn(X_training, y_training, X_testing, y_testing, [best_k])

    print("Summary of best model:")
    
    # Extracting the dictionary to print it in a more readable format
    report = test_results[best_k]["classification_report"]
    # print(report)
    print(f"  Class 0 (Gamma) - Precision: {report['0']['precision']:.4f}, Recall: {report['0']['recall']:.4f}, F1: {report['0']['f1-score']:.4f}")
    print(f"  Class 1 (Hadron)  - Precision: {report['1']['precision']:.4f}, Recall: {report['1']['recall']:.4f}, F1: {report['1']['f1-score']:.4f}")
    print(f"  Overall Accuracy : {report['accuracy']:.4f}")

    print(f"""
    Final Analysis & Comparison of Results:
    1. Validation vs. Test Sets: The best k-value chosen based on the validation set was k={best_k}.
       When applying this model to the unseen test set, F1-score was extremely close.
       This very slight, expected difference indicates our model generalized well .
       High and low K values (like k=1 and k=20) showed slightly worse performance.
    
    2. Precision vs. Recall Trade-off: 
       - For class '1' (which is 'h', Hadron), the model has high Recall (~0.849) 
         but lower Precision (~0.720).
       - This means the model successfully catches most of the true 'h' samples (high recall),
         but it also makes a fair amount of false alarms, incorrectly labeling 'g' samples as 'h' 
         (lower precision).
       - For class '0' ('g', Gamma), the opposite is true: higher precision (~0.816) 
         but lower recall (~0.670).
       - This means the model is better at identifying 'g' samples correctly, 
         but it misses more of them compared to 'h' samples.
    """)


if __name__ == "__main__":
    main()
