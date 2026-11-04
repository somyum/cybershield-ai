import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

COLS = ["duration","protocol_type","service","flag","src_bytes","dst_bytes","land",
"wrong_fragment","urgent","hot","num_failed_logins","logged_in","num_compromised",
"root_shell","su_attempted","num_root","num_file_creations","num_shells",
"num_access_files","num_outbound_cmds","is_host_login","is_guest_login","count",
"srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate",
"same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count",
"dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate",
"dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate",
"dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate",
"label","difficulty"]

ATTACK_MAP = {"normal":"normal",
 **dict.fromkeys(["neptune","smurf","back","teardrop","pod","land","apache2","udpstorm","processtable","mailbomb"],"DoS"),
 **dict.fromkeys(["satan","ipsweep","nmap","portsweep","mscan","saint"],"Probe"),
 **dict.fromkeys(["guess_passwd","ftp_write","imap","phf","multihop","warezmaster","warezclient","spy","xlock","xsnoop","snmpguess","snmpgetattack","httptunnel","sendmail","named","worm"],"R2L"),
 **dict.fromkeys(["buffer_overflow","loadmodule","rootkit","perl","sqlattack","xterm","ps"],"U2R")}

def load(path):
    df = pd.read_csv(path, names=COLS)
    df["category"] = df["label"].map(ATTACK_MAP).fillna("other")
    return df.drop(columns=["label","difficulty"])

train, test = load("data/KDDTrain+.txt"), load("data/KDDTest+.txt")
X_train, y_train = train.drop(columns="category"), train["category"]
X_test, y_test = test.drop(columns="category"), test["category"]

cat_cols = ["protocol_type","service","flag"]
pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)],
                        remainder="passthrough")
model = Pipeline([("pre", pre),
                  ("clf", RandomForestClassifier(n_estimators=200, n_jobs=-1,
                                                 class_weight="balanced", random_state=42))])

model.fit(X_train, y_train)
print(classification_report(y_test, model.predict(X_test)))
joblib.dump(model, "models/cybershield_rf.joblib")
print("Saved model.")