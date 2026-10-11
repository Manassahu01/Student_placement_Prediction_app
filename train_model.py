"""Train the placement decision tree and write the API model artifacts."""

from pathlib import Path
from typing import cast

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "student_performance_dataset.csv"
MODEL_PATH = BASE_DIR / "placement_model.joblib"
COLUMNS_PATH = BASE_DIR / "columns.joblib"

BRANCHES = ["CIVIL", "COMMERCE", "CSE", "ECE", "IT", "MECHANICAL", "UNKNOWN"]
NUMERIC_FEATURES = [
    "study_hours",
    "attendance",
    "sleep_hours",
    "internet_usage",
    "assignments_completed",
    "previous_score",
    "exam_score",
]
FEATURE_COLUMNS = [
    "gender",
    *NUMERIC_FEATURES[:-1],
    "extracurricular",
    "exam_score",
    *(f"branch_{branch}" for branch in BRANCHES),
]


def prepare_data(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    data = pd.read_csv(path)
    required = {
        "student_id", "gender", "branch", "study_hours", "attendance",
        "sleep_hours", "internet_usage", "assignments_completed",
        "previous_score", "extracurricular", "exam_score", "placement_status",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(sorted(missing))}")

    data = data.drop_duplicates().drop(columns="student_id").copy()
    data["gender"] = (
        data["gender"].astype("string").str.strip().str.lower()
        .map({"female": 0, "f": 0, "male": 1, "m": 1})
    )
    data["extracurricular"] = (
        data["extracurricular"].astype("string").str.strip().str.lower()
        .map({"no": 0, "yes": 1})
    )
    data["placement_status"] = (
        data["placement_status"].astype("string").str.strip().str.lower()
        .map({"not placed": 0, "placed": 1})
    )
    if data["gender"].isna().any():
        raise ValueError("Dataset contains an unsupported or missing gender value")
    if data["extracurricular"].isna().any():
        raise ValueError("Dataset contains an unsupported or missing extracurricular value")
    if data["placement_status"].isna().any():
        raise ValueError("Dataset contains an unsupported or missing placement_status value")
    data["branch"] = (
        data["branch"].astype("string").str.strip().str.upper()
        .where(lambda values: values.isin(BRANCHES[:-1]), "UNKNOWN")
        .fillna("UNKNOWN")
    )

    # Bound implausible daily-hour values using the API's accepted ranges.
    data["study_hours"] = data["study_hours"].clip(0, 12)
    data["sleep_hours"] = data["sleep_hours"].clip(3, 12)
    data["internet_usage"] = data["internet_usage"].clip(0, 12)

    data = data.dropna(subset=["placement_status"])
    branch_dummies = pd.get_dummies(data.pop("branch"), prefix="branch", dtype=int)
    X = pd.concat([data.drop(columns="placement_status"), branch_dummies], axis=1)
    X = X.reindex(columns=FEATURE_COLUMNS, fill_value=0)
    y = data["placement_status"].astype(int)
    return X, y


def main() -> None:
    X, y = prepare_data(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    medians = X_train.median()
    X_train = X_train.fillna(medians)
    X_test = X_test.fillna(medians)

    search = RandomizedSearchCV(
        estimator=DecisionTreeClassifier(random_state=42),
        param_distributions={
            "max_depth": [3, 5, 7, 10],
            "min_samples_leaf": [1, 2, 5, 10],
            "criterion": ["gini", "entropy"],
        },
        n_iter=10,
        cv=5,
        scoring="accuracy",
        random_state=42,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    model = cast(DecisionTreeClassifier, search.best_estimator_)

    y_pred = model.predict(X_test)
    print(f"Best parameters: {search.best_params_}")
    print(f"Train accuracy: {accuracy_score(y_train, model.predict(X_train)):.4f}")
    print(f"Test accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    joblib.dump(model, MODEL_PATH)
    joblib.dump(FEATURE_COLUMNS, COLUMNS_PATH)
    print(f"Saved model: {MODEL_PATH.name}")
    print(f"Saved features: {COLUMNS_PATH.name}")


if __name__ == "__main__":
    main()
