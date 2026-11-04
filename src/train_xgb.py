import joblib
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from data import load

train, test = load("data/KDDTrain+.txt"), load("data/KDDTest+.txt")
X_train, X_test = train.drop(columns="category"), test.drop(columns="category")
cat_cols = ["protocol_type", "service", "flag"]

def make_pipe():
    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)],
                            remainder="passthrough")
    clf = XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.1,
                        tree_method="hist", n_jobs=-1, random_state=42)
    return Pipeline([("pre", pre), ("clf", clf)])

# 1) Multi-class (attack category), rare classes up-weighted
le = LabelEncoder().fit(train["category"])
y_tr, y_te = le.transform(train["category"]), le.transform(test["category"])
multi = make_pipe()
multi.fit(X_train, y_tr, clf__sample_weight=compute_sample_weight("balanced", y_tr))
print("=== MULTI-CLASS ===")
print(classification_report(y_te, multi.predict(X_test), target_names=le.classes_))
joblib.dump((multi, le), "models/cybershield_xgb_multi.joblib")

# 2) Binary (attack vs normal)
yb_tr = (train["category"] != "normal").astype(int)
yb_te = (test["category"] != "normal").astype(int)
binary = make_pipe()
binary.fit(X_train, yb_tr)
print("=== BINARY ===")
print(classification_report(yb_te, binary.predict(X_test), target_names=["normal", "attack"]))
joblib.dump(binary, "models/cybershield_xgb_binary.joblib")