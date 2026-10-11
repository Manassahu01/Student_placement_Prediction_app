"""FastAPI service for the student placement decision-tree model."""

from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "placement_model.joblib"
COLUMNS_PATH = BASE_DIR / "columns.joblib"


def load_artifacts():
    """Load and check the model files when the API module starts."""
    missing = [path.name for path in (MODEL_PATH, COLUMNS_PATH) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing model artifact(s): {', '.join(missing)}. "
            "Run the final save cells in Decision_Tree.ipynb first."
        )

    model = joblib.load(MODEL_PATH)
    columns = joblib.load(COLUMNS_PATH)
    if not isinstance(columns, (list, tuple)) or not columns:
        raise ValueError("columns.joblib must contain a non-empty list of feature names")
    if any(not isinstance(column, str) for column in columns) or len(set(columns)) != len(columns):
        raise ValueError("columns.joblib must contain unique string feature names")
    if getattr(model, "n_features_in_", None) != len(columns):
        raise ValueError(
            "The model and columns.joblib disagree about the number of input features. "
            "Run train_model.py or the final save cells in Decision_Tree.ipynb again."
        )
    model_columns = getattr(model, "feature_names_in_", None)
    if model_columns is not None and list(model_columns) != list(columns):
        raise ValueError(
            "The feature order in columns.joblib does not match the trained model. "
            "Run train_model.py or the final save cells in Decision_Tree.ipynb again."
        )
    if not {0, 1}.issubset(set(getattr(model, "classes_", []))):
        raise ValueError("The model must have both placement classes 0 and 1")
    return model, list(columns)


model, columns = load_artifacts()
branch_columns = {
    column.removeprefix("branch_"): column
    for column in columns
    if column.startswith("branch_")
}

app = FastAPI(
    title="Student Placement Prediction API",
    description="Predict a student's placement status with the trained decision tree.",
    version="1.0.0",
)


class StudentData(BaseModel):
    """Student inputs; student_id is accepted for convenience but not used."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    student_id: int | None = None
    gender: str = Field(min_length=1)
    branch: str = Field(min_length=1)
    study_hours: float = Field(ge=0, le=12)
    attendance: float = Field(ge=0, le=100)
    sleep_hours: float = Field(ge=3, le=12)
    internet_usage: float = Field(ge=0, le=12)
    assignments_completed: int = Field(ge=0, le=20)
    previous_score: float = Field(ge=0, le=100)
    extracurricular: str = Field(min_length=1)
    exam_score: float = Field(ge=0, le=100)


@app.get("/")
def read_root():
    return FileResponse(BASE_DIR / "templates" / "index.html", media_type="text/html")


@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": True, "feature_count": len(columns)}


@app.post("/predict")
def predict_placement(student: StudentData):
    """Return the predicted class and its model probability."""
    gender = student.gender.strip().lower()
    gender_values = {"male": 1, "m": 1, "female": 0, "f": 0}
    if gender not in gender_values:
        raise HTTPException(status_code=422, detail="gender must be Male/Female or M/F")
    gender_value = gender_values[gender]

    extracurricular = student.extracurricular.strip().lower()
    if extracurricular not in {"yes", "no"}:
        raise HTTPException(status_code=422, detail="extracurricular must be Yes or No")
    extracurricular_value = int(extracurricular == "yes")

    branch = student.branch.strip().upper()
    if not branch:
        raise HTTPException(status_code=422, detail="branch must not be empty")
    branch_feature = branch_columns.get(branch, branch_columns.get("UNKNOWN"))
    if branch_feature is None and branch not in branch_columns:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported branch '{student.branch}'. Allowed branches: "
            f"{', '.join(sorted(branch_columns))}",
        )

    values = {
        "gender": gender_value,
        "study_hours": student.study_hours,
        "attendance": student.attendance,
        "sleep_hours": student.sleep_hours,
        "internet_usage": student.internet_usage,
        "assignments_completed": student.assignments_completed,
        "previous_score": student.previous_score,
        "extracurricular": extracurricular_value,
        "exam_score": student.exam_score,
        **{
            column: int(column == branch_feature)
            for column in columns
            if column.startswith("branch_")
        },
    }
    try:
        features = pd.DataFrame([ [values[column] for column in columns] ], columns=columns)
    except KeyError as error:
        raise RuntimeError(f"Unknown model feature in columns.joblib: {error}") from error

    prediction = int(model.predict(features)[0])
    placed_class_index = list(model.classes_).index(1)
    placement_probability = float(model.predict_proba(features)[0][placed_class_index])
    return {
        "placement_status": "Placed" if prediction == 1 else "Not Placed",
        "placement_probability": round(placement_probability, 4),
    }
