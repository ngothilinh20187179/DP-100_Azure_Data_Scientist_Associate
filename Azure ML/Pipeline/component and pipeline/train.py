# train.py (đặt ở thư mục gốc hoặc /src nếu bạn điều chỉnh path trong run.py)
import mlflow
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score # Thêm accuracy_score
import matplotlib.pyplot as plt
import os # Thêm import os

def main(args):
    df = get_data(args.training_data)
    X_train, X_test, y_train, y_test = split_data(df)
    model = train_model(args.reg_rate, X_train, y_train)
    eval_model(model, X_test, y_test)

def get_data(path):
    print(f"Reading data from: {path}")
    # Đảm bảo đường dẫn là một file nếu type=uri_file
    # Nếu path là một thư mục, bạn cần tìm file trong đó.
    # Với AssetTypes.URI_FILE, path sẽ trỏ trực tiếp đến file.
    df = pd.read_csv(path)
    return df

def split_data(df):
    X, y = df[['Pregnancies','PlasmaGlucose','DiastolicBloodPressure','TricepsThickness',
    'SerumInsulin','BMI','DiabetesPedigree','Age']].values, df['Diabetic'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)
    return X_train, X_test, y_train, y_test

def train_model(reg_rate, X_train, y_train):
    mlflow.log_param("Regularization rate", reg_rate)
    model = LogisticRegression(C=1/reg_rate, solver="liblinear", random_state=0).fit(X_train, y_train) # Thêm random_state
    return model

def eval_model(model, X_test, y_test):
    y_hat = model.predict(X_test)
    acc = accuracy_score(y_test, y_hat)
    mlflow.log_metric("accuracy_score", acc) 

    y_scores = model.predict_proba(X_test)
    auc = roc_auc_score(y_test,y_scores[:,1])
    mlflow.log_metric("AUC", auc)

    fpr, tpr, thresholds = roc_curve(y_test, y_scores[:,1])
    
    output_dir = './outputs'
    os.makedirs(output_dir, exist_ok=True)
    
    fig = plt.figure(figsize=(6, 4))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    roc_curve_path = os.path.join(output_dir, "ROC-Curve.png")
    plt.savefig(roc_curve_path)
    mlflow.log_artifact(roc_curve_path)
    plt.close(fig) # Đóng figure để tránh chiếm bộ nhớ

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_data", dest='training_data',
                        type=str, help="Path to training data file.")
    parser.add_argument("--reg_rate", dest='reg_rate',
                        type=float, default=0.01, help="Regularization rate (inverse of C).")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    mlflow.autolog() 
    main(args)