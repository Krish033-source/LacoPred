from flask import Flask, render_template, request, session
import pandas as pd
import matplotlib.pyplot as plt

from model import (
    predict_price,
    region_insight,
    price_hike_prediction,
    ai_budget_advisor
)

app = Flask(__name__)
app.secret_key = "lacopred_secret_key"


# --------- PRICE TREND GRAPH ---------
def generate_price_trend(region):
    data = pd.read_csv("region_price_history.csv")

    years = data["Year"]
    prices = data[region]

    plt.figure(figsize=(5, 3))
    plt.plot(years, prices, marker="o")
    plt.title(f"{region} Region Price Trend")
    plt.xlabel("Year")
    plt.ylabel("Price (₹ / sq ft)")
    plt.grid(True)

    path = f"static/{region}_trend.png"
    plt.savefig(path)
    plt.close()
    return path


# --------- MAIN ROUTE ---------
@app.route("/", methods=["GET", "POST"])
def index():

    # Initialize session storage
    session.setdefault("predictor", {})
    session.setdefault("advisor", {})
    session.setdefault("predictor_history", [])
    session.setdefault("advisor_history", [])

    if request.method == "POST":
        form_type = request.form.get("form_type")

        # --------- RESET ---------
        if form_type == "reset":
            session.clear()
            return render_template(
                "index.html",
                predictor={},
                advisor={},
                predictor_history=[],
                advisor_history=[]
            )

        # --------- LAND PRICE PREDICTOR ---------
        if form_type == "predictor":
            try:
                region = request.form["region"]
                area = float(request.form["area"])

                if area <= 0:
                    raise ValueError

                result = {
                    "region": region,
                    "area": area,
                    "price": predict_price(region, area),
                    "insight": region_insight(region),
                    "hike": price_hike_prediction(region),
                    "graph": generate_price_trend(region)
                }

                session["predictor"] = result
                session["predictor_history"].append(result)

            except Exception as e:
                print("Predictor error:", e)

        # --------- AI BUDGET ADVISOR ---------
        elif form_type == "advisor":
            try:
                budget = int(request.form["budget"])

                recommendation = ai_budget_advisor(budget)

                years = [2020, 2021, 2022, 2023, 2024]
                prices = recommendation["growth"]

                plt.figure(figsize=(5, 3))
                plt.plot(years, prices, marker="o")
                plt.title(f"{recommendation['region']} Growth Trend")
                plt.xlabel("Year")
                plt.ylabel("Price (₹ / sq ft)")
                plt.grid(True)

                graph_path = f"static/advisor_{recommendation['region']}.png"
                plt.savefig(graph_path)
                plt.close()

                result = {
                    "budget": budget,
                    "recommendation": recommendation,
                    "graph": graph_path
                }

                session["advisor"] = result
                session["advisor_history"].append(result)

            except Exception as e:
                print("Advisor error:", e)

    return render_template(
        "index.html",
        predictor=session.get("predictor"),
        advisor=session.get("advisor"),
        predictor_history=session.get("predictor_history"),
        advisor_history=session.get("advisor_history")
    )


# --------- RUN SERVER ---------
if __name__ == "__main__":
    app.run(debug=True)