# Academic Advisor (CSC 303)

A hybrid academic advisor: a **scikit-learn logistic regression model** (saved as `.pkl`) predicts your academic standing
(Excellent, Good, At Risk, Critical) and a **rule engine** turns it into a personal action plan.
It also includes a 5.0-scale GPA calculator. Served with **Flask**.

```
academic-advisor/
├── app.py                # Flask web app + JSON API
├── advisor_engine.py     # data generator, model wrapper, advice rules, GPA maths
├── train_model.py        # local training (same pipeline as the Colab notebook)
├── models/advisor_model.pkl
├── templates/ static/    # web interface
├── tests/                # pytest unit tests
├── .vscode/              # Python-optimised VS Code settings, debug config
└── .github/workflows/ci.yml
```

## 1. Run it in VS Code
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows   (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
python app.py                   # open http://127.0.0.1:5000
pytest                          # run the tests
```
Install the recommended extensions when VS Code prompts you (Python, Pylance, Black, Jupyter, GitLens).
Press F5 to run with the debugger.

## 2. Train in Google Colab
Upload `advisor_model_training.ipynb` to https://colab.research.google.com, choose Runtime > Run all,
and the last cell downloads a new `advisor_model.pkl`. Put it in `models/` (replace the old one) and restart the app.

## 3. Git and GitHub
```bash
git --version                   # confirm Git is installed (https://git-scm.com/downloads)
git init
git add .
git commit -m "Initial commit: academic advisor"
git branch -M main
# create an empty repo named academic-advisor on github.com, then:
git remote add origin https://github.com/<your-username>/academic-advisor.git
git push -u origin main
```
Or with the GitHub CLI (`gh --version` to confirm it is installed): `gh auth login` then
`gh repo create academic-advisor --public --source=. --push`.
Work on features in branches (`git checkout -b feature/gpa-calculator`) and open pull requests;
the CI workflow will train and test on every push.

## 4. Deploy (optional)
`gunicorn app:app` works on Render, Railway or PythonAnywhere.

## Notes
The training data is synthetic. Replace `generate_dataset()` with real anonymised student records
for a trustworthy model. If you see a scikit-learn version warning, rerun `python train_model.py`.
