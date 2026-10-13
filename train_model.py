"""Train locally (same pipeline as the Colab notebook) and write models/advisor_model.pkl."""
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from advisor_engine import BASE_DIR, FEATURES, MODEL_PATH, generate_dataset


def main():
    df = generate_dataset()
    df.to_csv(BASE_DIR / "data" / "students.csv", index=False)
    X, y = df[FEATURES], df["standing"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    model = Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, max_iter=2000)),
    ])
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    acc = accuracy_score(y_te, pred)
    print(classification_report(y_te, pred))
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES, "accuracy": acc}, MODEL_PATH)
    print(f"Saved {MODEL_PATH}  (accuracy {acc:.3f})")


if __name__ == "__main__":
    main()
