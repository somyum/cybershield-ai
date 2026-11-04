from pathlib import Path
from datetime import datetime, timezone
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from starlette.responses import FileResponse
from data import load, COLS

ROOT = Path(__file__).resolve().parent.parent
FEATURES = [c for c in COLS if c not in ("label", "difficulty")]
THRESHOLD = 0.1
SEVERITY = {"U2R": "critical", "R2L": "high", "DoS": "high",
            "Probe": "medium", "Unclassified": "medium", "normal": "none"}

binary = joblib.load(ROOT / "models/cybershield_xgb_binary.joblib")
multi, le = joblib.load(ROOT / "models/cybershield_xgb_multi.joblib")
test_df = load(str(ROOT / "data/KDDTest+.txt"))

alerts, scanned = [], 0

app = FastAPI(title="CyberShield AI")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


EXPLAIN = {
    "DoS": "Traffic pattern matches a denial-of-service attack: a burst of connections aimed at overwhelming the {service} service. Recommended: rate-limit the source and check server load.",
    "Probe": "Looks like reconnaissance: scanning {service}/{protocol} to map the network before an attack. Recommended: block the scanning source and review exposed ports.",
    "R2L": "Possible remote intrusion: an outside host trying to gain access, for example through password guessing. Recommended: check login attempts and enforce lockouts.",
    "U2R": "Possible privilege escalation: a local user trying to gain admin rights. Recommended: isolate the host and audit privileged accounts.",
    "Unclassified": "The detector flagged this connection as malicious, but the attack type is unclear. Recommended: manual review.",
}


def score(df: pd.DataFrame):
    global scanned
    X = df[FEATURES]
    probs = binary.predict_proba(X)[:, 1]
    cats = le.inverse_transform(multi.predict(X))
    results = []
    for (_, row), prob, cat in zip(df.iterrows(), probs, cats):
        is_attack = bool(prob >= THRESHOLD)
        if not is_attack:
            category = "normal"
        elif cat == "normal":
            category = "Unclassified"
        else:
            category = cat
        res ={
            "explanation": EXPLAIN.get(category, "").format(service=row["service"], protocol=row["protocol_type"]),
            "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "is_attack": is_attack,
            "attack_probability": round(float(prob), 3),
            "category": category,
            "severity": SEVERITY[category],
            "protocol": row["protocol_type"],
            "service": row["service"],
            "src_bytes": int(row["src_bytes"]),
            "dst_bytes": int(row["dst_bytes"]),
        }
        results.append(res)
        if is_attack:
            alerts.append(res)
    scanned += len(results)
    return results


@app.get("/health")
def health():
    return {"status": "ok", "threshold": THRESHOLD}


@app.post("/predict")
def predict(record: dict):
    missing = [f for f in FEATURES if f not in record]
    if missing:
        raise HTTPException(422, f"Missing features: {missing[:5]}...")
    return score(pd.DataFrame([record]))[0]


@app.post("/predict-batch")
def predict_batch(records: list[dict]):
    return score(pd.DataFrame(records))


@app.get("/simulate")
def simulate(n: int = 10):
    """Replay random real test traffic - this powers the live demo."""
    sample = test_df.sample(min(n, 100))
    results = score(sample)
    for r, truth in zip(results, sample["category"]):
        r["actual"] = truth
    return results


@app.get("/alerts")
def get_alerts(limit: int = 50):
    return alerts[-limit:][::-1]


@app.get("/stats")
def stats():
    by_cat = {}
    for a in alerts:
        by_cat[a["category"]] = by_cat.get(a["category"], 0) + 1
    return {"scanned": scanned, "alerts": len(alerts), "by_category": by_cat}



    from fastapi.responses import FileResponse

@app.get("/")
def dashboard():
    return FileResponse(ROOT / "src/dashboard.html")
@app.get("/simulate-attack")
def simulate_attack(n: int = 10):
    """Replay only real attack traffic - for the demo button."""
    sample = test_df[test_df["category"] != "normal"].sample(min(n, 50))
    results = score(sample)
    for r, truth in zip(results, sample["category"]):
        r["actual"] = truth
    return results