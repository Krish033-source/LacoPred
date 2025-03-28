import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder

# Load the dataset
df = pd.read_csv("land_price_data.csv")

# Encode categorical 'Region' column
encoder = OneHotEncoder()
encoded_region = encoder.fit_transform(df[['Region']]).toarray()
region_labels = encoder.get_feature_names_out(['Region'])
df_encoded = pd.DataFrame(encoded_region, columns=region_labels)

# Merge encoded data with original dataframe
df = pd.concat([df, df_encoded], axis=1)
df.drop(columns=['Region'], inplace=True)  # Drop original 'Region' column

# Define features (X) and target variable (y)
X = df.drop(columns=['Price (₹)'])
y = df['Price (₹)']

# Split data into training & testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Get user input
region_input = input("Enter the Region (North/South/East/West): ").strip().capitalize()
area_input = float(input("Enter the Land Area (sq ft): "))

# Validate region input
if region_input not in ["North", "South", "East", "West"]:
    print("❌ Invalid region! Please enter North, South, East, or West.")
else:
    # Create input data
    region_encoded = encoder.transform([[region_input]]).toarray()[0]
    input_data = pd.DataFrame([[area_input] + list(region_encoded)], columns=X.columns)

    # Predict land price
    predicted_price = model.predict(input_data)[0]
    print(f"✅ Predicted Land Price for {area_input} sq ft in {region_input}: ₹{round(predicted_price, 2)}")
