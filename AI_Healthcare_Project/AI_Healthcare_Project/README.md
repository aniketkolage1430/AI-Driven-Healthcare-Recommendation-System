# AI-Driven Healthcare Recommendation System

A complete Flask + Machine Learning healthcare dashboard that predicts possible diseases from symptoms and displays descriptions, precautions, medicines, diet plans, workouts, confidence scores, charts, voice input, PDF export, and local search history.

> Disclaimer: This project is for education, portfolio, viva, and internship showcase use. It is not a replacement for professional medical diagnosis.

## Project Structure

```text
AI_Healthcare_Project/
├── app.py
├── requirements.txt
├── README.md
├── models/
│   └── health_model.pkl
├── datasets/
│   ├── Training.csv
│   ├── description.csv
│   ├── precautions_df.csv
│   ├── medications.csv
│   ├── diets.csv
│   └── workout_df.csv
├── static/
│   ├── css/style.css
│   ├── js/script.js
│   └── images/
└── templates/
    └── index.html
```

## How It Works

- Loads `Training.csv`.
- Trains five models:
  - RandomForestClassifier
  - GradientBoostingClassifier
  - LogisticRegression
  - KNeighborsClassifier
  - SVC
- Compares accuracy and automatically saves the best model to `models/health_model.pkl`.
- Uses a hybrid prediction engine:
  - ML probability
  - rule-based symptom matching
  - synonym mapping
  - smart fallback scoring
- Returns top 5 disease predictions with percentages.
- Loads recommendation data from description, precaution, medication, diet, and workout CSV files.

## Run In PyCharm

1. Open PyCharm.
2. Choose **File > Open**.
3. Select the `AI_Healthcare_Project` folder.
4. Open the PyCharm terminal.
5. Run:

```bash
pip install -r requirements.txt
python app.py
```

6. Open:

```text
http://127.0.0.1:5000
```

## Sample Symptoms For Testing

```text
fever, chills, sweating
cough, high fever, breathlessness
heart burn, vomiting, stomach pain
skin rash, itching, nodal skin eruptions
headache, nausea, pain behind the eyes
burning urine, bladder discomfort
body pain, weakness, fatigue
```

## Deployment

### Render / Railway / Heroku Style

1. Push this folder to GitHub.
2. Create a new Python web service.
3. Install command:

```bash
pip install -r requirements.txt
```

4. Start command:

```bash
gunicorn app:app
```

5. Make sure the `datasets/` and `models/` folders are committed.

## Notes

- The saved model is generated from your dataset.
- If you delete `models/health_model.pkl`, the app will train again automatically on next run.
- PDF export happens in the browser using the print dialog.
