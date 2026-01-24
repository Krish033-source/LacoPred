import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder

# Load dataset
df = pd.read_csv("land_price_data.csv").dropna().reset_index(drop=True)

# Encode Region
encoder = OneHotEncoder(handle_unknown="ignore")
encoded = encoder.fit_transform(df[['Region']]).toarray()
labels = encoder.get_feature_names_out(['Region'])

encoded_df = pd.DataFrame(encoded, columns=labels)
df = pd.concat([df, encoded_df], axis=1)
df.drop(columns=['Region'], inplace=True)

X = df.drop(columns=['Price (₹)'])
y = df['Price (₹)']

X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)


def predict_price(region, area):
    region_encoded = encoder.transform([[region]]).toarray()[0]
    input_data = pd.DataFrame([[area] + list(region_encoded)], columns=X.columns)
    return round(model.predict(input_data)[0], 2)


def region_insight(region):
    insights = {
        "North": "Stable growth driven by infrastructure expansion.",
        "South": "Strong long-term appreciation due to IT and urban growth.",
        "East": "Emerging region with affordable pricing and future potential.",
        "West": "High demand region with strong commercial development."
    }
    return insights.get(region, "No insight available.")


def price_hike_prediction(region):
    forecast = {
        "North": "Moderate growth expected in 12–18 months.",
        "South": "Consistent growth expected over the next 2–3 years.",
        "East": "Gradual appreciation expected in the next 2 years.",
        "West": "High probability of short-term price increase."
    }
    return forecast.get(region, "No forecast available.")


def ai_budget_advisor(budget):
    if budget < 5_000_000:
        return {
            "region": "East",
            "area": "1200 – 1800 sq ft",
            "reason": "Affordable entry pricing with improving infrastructure.",
            "compare": "Cheaper than North & West, larger plots available.",
            "growth": [2500, 2700, 2900, 3100, 3400]
        }
    elif budget < 20_000_000:
        return {
            "region": "South",
            "area": "1000 – 1500 sq ft",
            "reason": "Strong demand driven by IT hubs and job growth.",
            "compare": "More stable than West, better appreciation than East.",
            "growth": [3000, 3300, 3600, 3900, 4200]
        }
    else:
        return {
            "region": "West",
            "area": "800 – 1200 sq ft",
            "reason": "Premium commercial activity and high ROI potential.",
            "compare": "Higher returns than North, but more volatile.",
            "growth": [3400, 3700, 4000, 4300, 4700]
        }
