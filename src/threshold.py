import joblib
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from data import load

test = load("data/KDDTest+.txt")
X_test = test.drop(columns="category")
y_true = (test["category"] != "normal").astype(int)

model = joblib.load("models/cybershield_xgb_binary.joblib")
proba = model.predict_proba(X_test)[:, 1]

print("thresh  attack_recall  attack_prec  f1    false_alarm_rate")
for t in [0.5, 0.4, 0.3, 0.2, 0.1, 0.05]:
    pred = (proba >= t).astype(int)
    fa = ((pred == 1) & (y_true == 0)).sum() / (y_true == 0).sum()
    print(f"{t:<7} {recall_score(y_true, pred):<14.2f} {precision_score(y_true, pred):<12.2f} "
          f"{f1_score(y_true, pred):<5.2f} {fa:.2f}")