# Student Placement Prediction

This project trains a decision-tree classifier from `student_performance_dataset.csv` and serves predictions through a FastAPI application.

## Project files

- `Decision_Tree.ipynb` — data exploration followed by the canonical training workflow in `train_model.py`.
- `train_model.py` — repeatable command-line training script. It writes the same two artifacts used by the API.
- `app.py` — prediction API.
- `placement_model.joblib` and `columns.joblib` — trained model and its ordered feature names.
- `student_performance_dataset.csv` — training data.
- `requirements.txt` — Python dependencies.

## Setup and run

Use Python 3.10 or newer and run these commands from the project directory:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

To retrain the model and regenerate both artifacts:

```powershell
python train_model.py
```

To start the API:

```powershell
python -m uvicorn app:app --reload
```

Open `http://127.0.0.1:8000/docs` for the interactive API documentation. `GET /health` reports whether the model artifacts loaded. `POST /predict` accepts a JSON student record and returns the predicted status and class probability.

Example request:

```json
{
  "gender": "Female",
  "branch": "CSE",
  "study_hours": 5.5,
  "attendance": 85,
  "sleep_hours": 7,
  "internet_usage": 3,
  "assignments_completed": 8,
  "previous_score": 78,
  "extracurricular": "Yes",
  "exam_score": 82
}
```

`student_id` is optional and is not used by the model. Gender accepts `Male`, `Female`, `M`, or `F`; extracurricular accepts `Yes` or `No`. Branch values are normalized to uppercase; any branch outside CIVIL, COMMERCE, CSE, ECE, IT, and MECHANICAL uses the model's `UNKNOWN` category. The daily-hour fields are limited to 0–12 for study and internet use, and 3–12 for sleep. Assignment count is limited to 0–20, matching the training data range.

`placement_probability` always means the model's probability for the `Placed` class, even when the predicted status is `Not Placed`.

## Notebook workflow

Open `Decision_Tree.ipynb` from this directory and run the cells from top to bottom. Its final training cell calls `train_model.py`, so the notebook and command-line workflow use identical cleaning, imputation, feature columns, tuning, and artifact paths. Both create `placement_model.joblib` and `columns.joblib`; the API requires both files.
