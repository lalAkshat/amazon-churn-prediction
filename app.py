from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

# Load files
model = pickle.load(open("churn_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
feature_order = pickle.load(open("feature_order.pkl", "rb"))

app = Flask(__name__)

# Home page (Web UI)
@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    probability = None
    risk = None

    if request.method == "POST":
        data = {
            "CreditScore": int(request.form["CreditScore"]),
            "Age": int(request.form["Age"]),
            "Tenure": int(request.form["Tenure"]),
            "Balance": float(request.form["Balance"]),
            "NumOfProducts": int(request.form["NumOfProducts"]),
            "HasCrCard": int(request.form["HasCrCard"]),
            "IsActiveMember": int(request.form["IsActiveMember"]),
            "EstimatedSalary": float(request.form["EstimatedSalary"]),
            "ZeroBalance": 1 if float(request.form["Balance"]) == 0 else 0,
            "ShortTenure": 1 if int(request.form["Tenure"]) <= 2 else 0,
            "HighValueCustomer": 1,
            "MultiProductCustomer": 1 if int(request.form["NumOfProducts"]) > 1 else 0,
            "Geography_Germany": 1 if request.form["Geography"] == "Germany" else 0,
            "Geography_Spain": 1 if request.form["Geography"] == "Spain" else 0,
            "Gender_Male": 1 if request.form["Gender"] == "Male" else 0,
            "AgeGroup_Mid": 1 if 30 <= int(request.form["Age"]) < 45 else 0,
            "AgeGroup_Senior": 1 if 45 <= int(request.form["Age"]) < 60 else 0,
            "AgeGroup_Old": 1 if int(request.form["Age"]) >= 60 else 0
        }

        df = pd.DataFrame([data])
        df = df.reindex(columns=feature_order, fill_value=0)

        scaled = scaler.transform(df)
        prediction = model.predict(scaled)[0]
        probability = round(model.predict_proba(scaled)[0][1], 3)

        risk = (
            "High Risk" if probability >= 0.7 else
            "Medium Risk" if probability >= 0.4 else
            "Low Risk"
        )

    return render_template(
        "index.html",
        prediction=prediction,
        probability=probability,
        risk=risk
    )

# API endpoint (optional)
@app.route("/predict", methods=["POST"])
def predict_api():
    data = request.json
    df = pd.DataFrame([data])
    df = df.reindex(columns=feature_order, fill_value=0)
    scaled = scaler.transform(df)

    prediction = model.predict(scaled)[0]
    probability = model.predict_proba(scaled)[0][1]

    return jsonify({
        "Churn_Prediction": int(prediction),
        "Churn_Probability": round(probability, 3),
        "Risk_Level": (
            "High Risk" if probability >= 0.7 else
            "Medium Risk" if probability >= 0.4 else
            "Low Risk"
        )
    })

if __name__ == "__main__":
    app.run(debug=True)