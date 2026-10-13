"""Core logic: data generation, prediction wrapper, rule-based advice, GPA maths.

Hybrid design
  * AI part   -> a scikit-learn classifier (loaded from models/advisor_model.pkl)
  * Rules part -> transparent, testable advice rules (standard software engineering)
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "advisor_model.pkl"

FEATURES = [
    "cgpa",            # cumulative GPA, 0-5 scale
    "semester_gpa",    # last semester GPA, 0-5 scale
    "attendance",      # % of classes attended
    "study_hours",     # self-study hours per week
    "assignment_rate", # % of assignments submitted
    "carryovers",      # number of outstanding failed courses
    "sleep_hours",     # average hours of sleep per night
    "level",           # 100, 200, 300, 400
]
CLASSES = ["Critical", "At Risk", "Good", "Excellent"]

FEATURE_LIMITS = {
    "cgpa": (0, 5), "semester_gpa": (0, 5), "attendance": (0, 100),
    "study_hours": (0, 80), "assignment_rate": (0, 100),
    "carryovers": (0, 20), "sleep_hours": (0, 14), "level": (100, 600),
}

GRADE_POINTS = {"A": 5, "B": 4, "C": 3, "D": 2, "E": 1, "F": 0}


# ----------------------------------------------------------------- data
def generate_dataset(n: int = 4000, seed: int = 42) -> pd.DataFrame:
    """Synthetic student records. Replace with real (anonymised) data when you have it."""
    rng = np.random.default_rng(seed)
    cgpa = np.clip(rng.normal(3.3, 0.9, n), 0.8, 5)
    semester_gpa = np.clip(cgpa + rng.normal(0, 0.6, n), 0, 5)
    attendance = np.clip(rng.normal(76 + (cgpa - 3.3) * 6, 14, n), 20, 100)
    study_hours = np.clip(rng.normal(12 + (cgpa - 3.3) * 4, 6, n), 0, 45)
    assignment_rate = np.clip(rng.normal(78 + (cgpa - 3.3) * 7, 15, n), 10, 100)
    carryovers = np.clip(np.round(rng.poisson(np.clip(3.2 - cgpa, 0.05, 3), n)), 0, 12)
    sleep_hours = np.clip(rng.normal(6.4, 1.3, n), 3, 10)
    level = rng.choice([100, 200, 300, 400], n)

    projected = (
        0.50 * cgpa + 0.30 * semester_gpa
        + 0.70 * (attendance - 75) / 25
        + 0.55 * (assignment_rate - 70) / 30
        + 0.40 * (np.minimum(study_hours, 25) - 10) / 15
        - 0.22 * carryovers
        - 0.12 * np.abs(sleep_hours - 7.5)
        + 0.75
        + rng.normal(0, 0.28, n)
    )
    projected = np.clip(projected, 0, 5)
    label = np.select(
        [projected >= 4.3, projected >= 3.4, projected >= 2.4],
        ["Excellent", "Good", "At Risk"], default="Critical",
    )
    df = pd.DataFrame({
        "cgpa": cgpa.round(2), "semester_gpa": semester_gpa.round(2),
        "attendance": attendance.round(0), "study_hours": study_hours.round(1),
        "assignment_rate": assignment_rate.round(0), "carryovers": carryovers.astype(int),
        "sleep_hours": sleep_hours.round(1), "level": level, "standing": label,
    })
    return df


# ----------------------------------------------------------- validation
class ValidationError(ValueError):
    pass


def validate_input(payload: dict) -> dict:
    clean = {}
    for name in FEATURES:
        if name not in payload or payload[name] in ("", None):
            raise ValidationError(f"Missing value: {name}")
        try:
            value = float(payload[name])
        except (TypeError, ValueError):
            raise ValidationError(f"{name} must be a number")
        lo, hi = FEATURE_LIMITS[name]
        if not lo <= value <= hi:
            raise ValidationError(f"{name} must be between {lo} and {hi}")
        clean[name] = value
    return clean


# ----------------------------------------------------------- AI model
_bundle = None


def load_bundle(path: Path = MODEL_PATH) -> dict:
    global _bundle
    if _bundle is None:
        _bundle = joblib.load(path)
    return _bundle


def predict(payload: dict) -> dict:
    data = validate_input(payload)
    bundle = load_bundle()
    row = pd.DataFrame([[data[f] for f in FEATURES]], columns=FEATURES)
    model = bundle["model"]
    proba = model.predict_proba(row)[0]
    classes = list(model.classes_)
    standing = classes[int(np.argmax(proba))]
    probs = {c: round(float(p), 4) for c, p in zip(classes, proba)}
    return {
        "standing": standing,
        "confidence": round(float(proba.max()), 4),
        "probabilities": {c: probs.get(c, 0.0) for c in CLASSES},
        "advice": build_advice(data, standing),
        "summary": SUMMARIES[standing],
    }


# ------------------------------------------------------- rule engine
SUMMARIES = {
    "Excellent": "You are on a strong track. Protect your routine and aim higher.",
    "Good": "You are doing well, and a few small changes could lift you a class.",
    "At Risk": "Your results could slip next semester. Act on the points below now.",
    "Critical": "Your standing is in danger. Please see your course adviser this week.",
}


def build_advice(d: dict, standing: str) -> list[dict]:
    tips = []

    def add(level, title, detail):
        tips.append({"level": level, "title": title, "detail": detail})

    if d["attendance"] < 70:
        add("high", "Attendance is below 70%",
            "Many departments bar students under 75% from exams. Prioritise showing up for every lecture and lab.")
    elif d["attendance"] < 85:
        add("medium", "Push attendance above 85%", "Consistent attendance is one of the strongest predictors of grades.")

    if d["carryovers"] >= 3:
        add("high", f"{int(d['carryovers'])} carry-over courses",
            "Register them first next semester and give them at least half of your study time.")
    elif d["carryovers"] >= 1:
        add("medium", "Clear your carry-over", "Retake it early, before new courses pile up.")

    if d["study_hours"] < 8:
        add("high", "Study time is very low", "Aim for at least 12 focused hours per week, split into 90-minute blocks.")
    elif d["study_hours"] < 14:
        add("medium", "Add a few study hours", "Moving from about 10 to 15 hours a week usually shows within one semester.")

    if d["assignment_rate"] < 70:
        add("high", "Missing assignments", "Continuous assessment carries marks you can win. Submit every assignment, even when imperfect.")

    if d["semester_gpa"] + 0.4 < d["cgpa"]:
        add("medium", "Your last semester dropped", "Find what changed (workload, health, finances) and talk to your adviser early.")
    elif d["semester_gpa"] > d["cgpa"] + 0.3:
        add("low", "Positive momentum", "Your last semester beat your CGPA. Keep the habits that worked.")

    if d["sleep_hours"] < 6:
        add("medium", "Sleep more", "Under 6 hours hurts memory and focus. Target 7 to 8.")

    needed = gpa_needed(d["cgpa"], d["level"])
    if needed:
        add("low", "Class-of-degree target", needed)

    if standing == "Excellent" and not tips:
        add("low", "Keep going", "Consider research, a project supervisor, or peer tutoring to build your profile.")
    order = {"high": 0, "medium": 1, "low": 2}
    return sorted(tips, key=lambda t: order[t["level"]])


def gpa_needed(cgpa: float, level: int) -> str | None:
    """Rough guide: GPA to hold for the rest of a 4-year degree to reach First Class (4.50)."""
    done = {100: 1, 200: 2, 300: 3, 400: 4}.get(int(level))
    if done is None or done >= 4:
        return None
    remaining = 4 - done
    req = (4.5 * 4 - cgpa * done) / remaining
    if req > 5:
        return "First Class (4.50) is now mathematically out of reach, so aim for Second Class Upper (3.50)."
    return f"To finish with First Class (4.50) you need about {req:.2f} per year for the remaining {remaining} year(s)."


# ---------------------------------------------------------- GPA maths
def calculate_gpa(courses: list[dict]) -> dict:
    """courses = [{"code": "CSC 303", "units": 3, "grade": "A"}, ...]"""
    total_units = total_points = 0
    for c in courses:
        grade = str(c.get("grade", "")).upper()
        if grade not in GRADE_POINTS:
            raise ValidationError(f"Invalid grade: {c.get('grade')}")
        try:
            units = float(c.get("units", 0))
        except (TypeError, ValueError):
            raise ValidationError("Units must be a number")
        if not 0 < units <= 10:
            raise ValidationError("Units must be between 1 and 10")
        total_units += units
        total_points += units * GRADE_POINTS[grade]
    if total_units == 0:
        raise ValidationError("Add at least one course")
    gpa = total_points / total_units
    return {"gpa": round(gpa, 2), "units": total_units, "class": degree_class(gpa)}


def degree_class(gpa: float) -> str:
    if gpa >= 4.5: return "First Class"
    if gpa >= 3.5: return "Second Class Upper"
    if gpa >= 2.4: return "Second Class Lower"
    if gpa >= 1.5: return "Third Class"
    return "Pass / Probation"
