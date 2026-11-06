# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from flask import Flask, render_template, request
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

gender_encoder = joblib.load("gender_encoder.pkl")

app = Flask(__name__)


# =====================================================
# LOAD DATASET
# =====================================================

df = pd.read_csv(r"D:\OneDrive\Desktop\covid 19\covid19_patient_outcome.csv")

print("Dataset loaded successfully")
print("Dataset shape:", df.shape)


# =====================================================
# REMOVE ID
# =====================================================

df = df.drop("Patient_ID", axis=1)


# =====================================================
# HANDLE MISSING VALUES
# =====================================================

numeric_columns = df.select_dtypes(
    include=np.number
).columns

for column in numeric_columns:

    if column != "Outcome":

        df[column] = df[column].fillna(
            df[column].median()
        )


categorical_columns = df.select_dtypes(
    include="object"
).columns

for column in categorical_columns:

    if column != "Outcome":

        df[column] = df[column].fillna(
            df[column].mode()[0]
        )


# =====================================================
# ENCODE GENDER
# =====================================================

gender_encoder = LabelEncoder()

df["Gender"] = gender_encoder.fit_transform(
    df["Gender"]
)


# =====================================================
# ENCODE OUTCOME
# =====================================================

outcome_encoder = LabelEncoder()

df["Outcome"] = outcome_encoder.fit_transform(
    df["Outcome"]
)


# =====================================================
# OUTLIER HANDLING
# =====================================================

outlier_columns = [
    "Age",
    "Oxygen_Level",
    "Respiratory_Rate",
    "Temperature"
]

for column in outlier_columns:

    Q1 = df[column].quantile(0.25)

    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR

    upper = Q3 + 1.5 * IQR

    df[column] = df[column].clip(
        lower,
        upper
    )


# =====================================================
# X AND Y
# =====================================================

X = df.drop(
    "Outcome",
    axis=1
)

y = df["Outcome"]


# =====================================================
# TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.50,

    random_state=42,

    stratify=y
)


# =====================================================
# RANDOM FOREST
# =====================================================

model = RandomForestClassifier(

    n_estimators=100,

    max_depth=10,

    min_samples_split=4,

    min_samples_leaf=2,

    max_features="sqrt",

    class_weight="balanced",

    random_state=42,

    n_jobs=-1
)


# =====================================================
# TRAIN
# =====================================================

model.fit(
    X_train,
    y_train
)


# =====================================================
# ACCURACY
# =====================================================

y_pred = model.predict(
    X_test
)

accuracy = accuracy_score(
    y_test,
    y_pred
)

accuracy_percentage = round(
    accuracy * 100,
    2
)

print(
    "Random Forest Accuracy:",
    accuracy_percentage,
    "%"
)


# =====================================================
# HOME PAGE
# =====================================================

@app.route("/")
def home():

    return render_template(
        "index.html",
        accuracy=accuracy_percentage
    )


# =====================================================
# PREDICTION
# =====================================================

@app.route(
    "/predict",
    methods=["POST"]
)

def predict():

    try:

        age = float(
            request.form["Age"]
        )

        gender = request.form["Gender"]
        gender_encoded = gender_encoder.transform([gender])[0]

        fever = int(
            request.form["Fever"]
        )

        cough = int(
            request.form["Cough"]
        )

        breathing = int(
            request.form["Breathing_Difficulty"]
        )

        diabetes = int(
            request.form["Diabetes"]
        )

        hypertension = int(
            request.form["Hypertension"]
        )

        heart_disease = int(
            request.form["Heart_Disease"]
        )

        oxygen = float(
            request.form["Oxygen_Level"]
        )

        respiratory = float(
            request.form["Respiratory_Rate"]
        )

        temperature = float(
            request.form["Temperature"]
        )

        hospitalized = int(
            request.form["Hospitalized"]
        )

        icu = int(
        request.form["ICU_Required"]
       )


        # =============================================
        # VALIDATION
        # =============================================

        if age < 18 or age > 120:

            raise ValueError(
                "Age must be between 18 and 120."
            )


        if oxygen < 50 or oxygen > 100:

            raise ValueError(
                "Oxygen level must be between 50 and 100."
            )


        if respiratory < 5 or respiratory > 60:

            raise ValueError(
                "Respiratory rate must be between 5 and 60."
            )


        if temperature < 30 or temperature > 45:

            raise ValueError(
                "Temperature must be between 30°C and 45°C."
            )


        # =============================================
        # ENCODE GENDER
        # =============================================

        gender_encoded = gender_encoder.transform(
            [gender]
        )[0]


        # =============================================
        # NEW PATIENT DATA
        # =============================================

        new_patient = pd.DataFrame({

            "Age": [age],

            "Gender": [gender_encoded],

            "Fever": [fever],

            "Cough": [cough],

            "Breathing_Difficulty": [
                breathing
            ],

            "Diabetes": [
                diabetes
            ],

            "Hypertension": [
                hypertension
            ],

            "Heart_Disease": [
                heart_disease
            ],

            "Oxygen_Level": [
                oxygen
            ],

            "Respiratory_Rate": [
                respiratory
            ],

            "Temperature": [
                temperature
            ],

            "Hospitalized": [
                hospitalized
            ],

            "ICU_Required": [
                icu
            ]

        })


        # =============================================
        # HANDLE OUTLIERS FOR NEW DATA
        # =============================================

        for column in outlier_columns:

            Q1 = X_train[column].quantile(
                0.25
            )

            Q3 = X_train[column].quantile(
                0.75
            )

            IQR = Q3 - Q1

            lower = Q1 - 1.5 * IQR

            upper = Q3 + 1.5 * IQR

            new_patient[column] = (
                new_patient[column]
                .clip(lower, upper)
            )


        # =============================================
        # PREDICTION
        # =============================================

        prediction = model.predict(
            new_patient
        )


        predicted_outcome = (
            outcome_encoder
            .inverse_transform(
                prediction
            )[0]
        )


        # =============================================
        # PROBABILITY
        # =============================================

        probabilities = model.predict_proba(
            new_patient
        )[0]

        confidence = round(
            max(probabilities) * 100,
            2
        )


        # =============================================
        # SEND RESULT TO HTML
        # =============================================

        return render_template(

            "index.html",

            prediction=predicted_outcome,

            confidence=confidence,

            accuracy=accuracy_percentage,

            patient=request.form

        )


    except Exception as error:

        return render_template(

            "index.html",

            error=str(error),

            accuracy=accuracy_percentage

        )


# =====================================================
# RUN FLASK
# =====================================================

if __name__ == "__main__":

    app.run(
        debug=True,
        use_reloader=False
    )
