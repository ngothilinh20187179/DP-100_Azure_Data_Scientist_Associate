import mlflow
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def main(args):
    # enable autologging
    mlflow.autolog()

    df = get_data(args.training_data)
    X_train, X_test, y_train, y_test = split_data(df)
    model = train_model(args.reg_rate, X_train, X_test, y_train, y_test)
    eval_model(model, X_test, y_test)

def get_data(path):
    df = pd.read_csv(path)
    return df

def split_data(df):
    X, y = df[['Pregnancies','PlasmaGlucose','DiastolicBloodPressure','TricepsThickness','SerumInsulin','BMI','DiabetesPedigree','Age']].values, df['Diabetic'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)
    return X_train, X_test, y_train, y_test

def train_model(reg_rate, X_train, X_test, y_train, y_test):
    model = LogisticRegression(C=1/reg_rate, solver="liblinear").fit(X_train, y_train)
    return model

def eval_model(model, X_test, y_test):
    y_hat = model.predict(X_test)
    acc = np.average(y_hat == y_test)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_data", dest='training_data', type=str)
    parser.add_argument("--reg_rate", dest='reg_rate', type=float, default=0.01)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # parse args
    args = parse_args()
    main(args)
    
    
# Specify the flavor with autologging
# You can use autologging, but still specify the flavor of the model. In the example, the model's flavor is scikit-learn.

# def main(args):
#     # enable autologging
#     mlflow.sklearn.autolog()
#     # ....

# # function that evaluates the model
# def eval_model(model, X_test, y_test):
#     # calculate accuracy
#     y_hat = model.predict(X_test)
#     acc = np.average(y_hat == y_test)
#     print('Accuracy:', acc)

#     # calculate AUC
#     y_scores = model.predict_proba(X_test)
#     auc = roc_auc_score(y_test,y_scores[:,1])
#     print('AUC: ' + str(auc))

#     # plot ROC curve
#     fpr, tpr, thresholds = roc_curve(y_test, y_scores[:,1])
#     fig = plt.figure(figsize=(6, 4))
#     # Plot the diagonal 50% line
#     plt.plot([0, 1], [0, 1], 'k--')
#     # Plot the FPR and TPR achieved by our model
#     plt.plot(fpr, tpr)
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('ROC Curve')
#     plt.savefig("ROC-Curve.png") 



# Customize the model with an inferred signature
# from mlflow.models.signature import infer_signature
# def main(args):
#     df = get_data(args.training_data)
#     X_train, X_test, y_train, y_test = split_data(df)
#     model = train_model(args.reg_rate, X_train, X_test, y_train, y_test)
#     y_hat = eval_model(model, X_test, y_test)

#     # create the signature by inferring it from the datasets
#     signature = infer_signature(X_train, y_hat)

#     # manually log the model
#     mlflow.sklearn.log_model(model, "model", signature=signature)

# def eval_model(model, X_test, y_test):
#     # ...
#     return y_hat


# Customize the model with a defined signature
# from mlflow.types.schema import Schema, ColSpec
# from mlflow.models.signature import ModelSignature
# from sklearn.metrics import roc_auc_score
# from sklearn.metrics import roc_curve
# import matplotlib.pyplot as plt

# def main(args):
#     df = get_data(args.training_data)
#     X_train, X_test, y_train, y_test = split_data(df)
#     model = train_model(args.reg_rate, X_train, X_test, y_train, y_test)
#     y_hat = eval_model(model, X_test, y_test)

#     # create the signature manually
#     input_schema = Schema([
#     ColSpec("integer", "Pregnancies"),
#     ColSpec("integer", "PlasmaGlucose"),
#     ColSpec("integer", "DiastolicBloodPressure"),
#     ColSpec("integer", "TricepsThickness"),
#     ColSpec("integer", "DiastolicBloodPressure"),
#     ColSpec("integer", "SerumInsulin"),
#     ColSpec("double", "BMI"),
#     ColSpec("double", "DiabetesPedigree"),
#     ColSpec("integer", "Age"),
#     ])
#     output_schema = Schema([ColSpec("boolean")])
#     # Create the signature object
#     signature = ModelSignature(inputs=input_schema, outputs=output_schema)
#     # manually log the model
#     mlflow.sklearn.log_model(model, "model", signature=signature)

# def eval_model(model, X_test, y_test):
#     # ...
#     return y_hat