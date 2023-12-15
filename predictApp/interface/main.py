import streamlit as st
import joblib
import pandas as pd
import os

st.set_page_config(
    page_title="Life Prediction",
    layout='wide'
)

#Random data for test
data = [
    {'Country': 'Nepal', 'DoB': 1963, 'Sex': 0, 'Socpro': 2, 'BMI': 24.320734, 'Diet_Hab': 2, 'Phys_act': 1, 'Drink_stat': 2, 'Smoking_stat': 3, 'Family_dis': 3, 'Chronic': 3},
    {'Country': 'Antigua and Barbuda', 'DoB': 1955, 'Sex': 0, 'Socpro': 2, 'BMI': 25.276657, 'Diet_Hab': 0, 'Phys_act': 4, 'Drink_stat': 2, 'Smoking_stat': 1, 'Family_dis': 1, 'Chronic': 3},
    {'Country': 'Paraguay', 'DoB': 1982, 'Sex': 1, 'Socpro': 1, 'BMI': 25.060033, 'Diet_Hab': 0, 'Phys_act': 1, 'Drink_stat': 0, 'Smoking_stat': 1, 'Family_dis': 1, 'Chronic': 0},
    {'Country': 'Austria', 'DoB': 1958, 'Sex': 1, 'Socpro': 0, 'BMI': 21.742619, 'Diet_Hab': 1, 'Phys_act': 4, 'Drink_stat': 1, 'Smoking_stat': 5, 'Family_dis': 2, 'Chronic': 1},
    {'Country': 'Gambia, The', 'DoB': 1994, 'Sex': 1, 'Socpro': 2, 'BMI': 37.386777, 'Diet_Hab': 0, 'Phys_act': 4, 'Drink_stat': 1, 'Smoking_stat': 1, 'Family_dis': 2, 'Chronic': 1},
    {'Country': 'Burundi', 'DoB': 1982, 'Sex': 0, 'Socpro': 1, 'BMI': 39.030496, 'Diet_Hab': 0, 'Phys_act': 4, 'Drink_stat': 3, 'Smoking_stat': 1, 'Family_dis': 3, 'Chronic': 0},
    {'Country': 'Bosnia and Herzegovina', 'DoB': 1978, 'Sex': 0, 'Socpro': 0, 'BMI': 39.041657, 'Diet_Hab': 0, 'Phys_act': 4, 'Drink_stat': 0, 'Smoking_stat': 3, 'Family_dis': 3, 'Chronic': 1},
    {'Country': 'Mozambique', 'DoB': 1955, 'Sex': 0, 'Socpro': 4, 'BMI': 31.055706, 'Diet_Hab': 3, 'Phys_act': 1, 'Drink_stat': 0, 'Smoking_stat': 4, 'Family_dis': 4, 'Chronic': 2},
    {'Country': 'Bahamas, The', 'DoB': 2007, 'Sex': 1, 'Socpro': 3, 'BMI': 22.617634, 'Diet_Hab': 1, 'Phys_act': 0, 'Drink_stat': 3, 'Smoking_stat': 5, 'Family_dis': 3, 'Chronic': 3},
    {'Country': 'Eswatini', 'DoB': 1949, 'Sex': 0, 'Socpro': 0, 'BMI': 47.380797, 'Diet_Hab': 3, 'Phys_act': 1, 'Drink_stat': 0, 'Smoking_stat': 4, 'Family_dis': 2, 'Chronic': 0},
]

df = pd.DataFrame(data)

data_array = df

#Loading model
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, '..', 'model', 'model.pkl')
preprocessor_path = os.path.join(script_dir, '..', 'model', 'preprocessor.joblib')

url_model = model_path
url_preprocessor = preprocessor_path

model = joblib.load(url_model)
preprocessor = joblib.load(url_preprocessor)

data_array_proproc = preprocessor.transform(data_array)

prediction = model.predict(data_array_proproc)

number = 3

def app():
    st.write("To be completed")
    st.write(f"Test data : {data[number]}")
    st.write(f"Life Expectancy Prediction: {round(prediction[number],1)} years")


if __name__ == "__main__":
    app()
