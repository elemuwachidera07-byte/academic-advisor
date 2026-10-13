import pytest

from advisor_engine import ValidationError, calculate_gpa, degree_class, predict
from app import app

STRONG = dict(cgpa=4.7, semester_gpa=4.8, attendance=96, study_hours=22,
              assignment_rate=98, carryovers=0, sleep_hours=7.5, level=300)
WEAK = dict(cgpa=1.6, semester_gpa=1.2, attendance=42, study_hours=3,
            assignment_rate=35, carryovers=6, sleep_hours=4.5, level=200)


def test_strong_student_is_not_flagged():
    assert predict(STRONG)["standing"] in ("Excellent", "Good")


def test_weak_student_is_flagged():
    assert predict(WEAK)["standing"] in ("Critical", "At Risk")


def test_probabilities_sum_to_one():
    assert sum(predict(STRONG)["probabilities"].values()) == pytest.approx(1, abs=0.01)


def test_validation_rejects_bad_input():
    with pytest.raises(ValidationError):
        predict({**STRONG, "cgpa": 9})


def test_gpa_calculation():
    r = calculate_gpa([{"units": 3, "grade": "A"}, {"units": 3, "grade": "C"}])
    assert r["gpa"] == 4.0 and r["class"] == "Second Class Upper"


def test_degree_class_boundaries():
    assert degree_class(4.5) == "First Class"
    assert degree_class(2.39) == "Third Class"


def test_api_endpoints():
    c = app.test_client()
    assert c.get("/health").status_code == 200
    assert c.get("/").status_code == 200
    assert c.post("/api/predict", json=STRONG).status_code == 200
    assert c.post("/api/predict", json={}).status_code == 400
