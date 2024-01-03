import streamlit as st
import joblib
import pandas as pd
import os

st.set_page_config(
    page_title="Life Prediction",
    layout='wide'
)

countries = ['Turkiye', 'Spain', 'India', 'Guyana', 'Israel', 'Costa Rica',
       'Russian Federation', 'Hungary', 'Jordan', 'Moldova', 'Brazil',
       'Malta', 'Bahamas, The', 'Ukraine', 'Switzerland', 'Norway',
       'Finland', 'Comoros', 'Japan', 'Gabon', 'Ghana', 'Philippines',
       'Congo, Rep.', 'Madagascar', 'Estonia', 'Belize', 'Kazakhstan',
       'Cameroon', 'Zimbabwe', 'Bhutan', 'South Africa', 'Eritrea',
       'Germany', 'Saudi Arabia', 'Kiribati', 'Seychelles', 'Singapore',
       'Togo', 'Denmark', 'Gambia, The', 'Sweden', 'Austria',
       'Kyrgyz Republic', 'Grenada', 'Brunei Darussalam', 'Greece',
       'Uruguay', 'Croatia', 'Romania', 'Central African Republic',
       'Algeria', 'Yemen, Rep.', 'Armenia',
       'St. Vincent and the Grenadines', 'Kenya', 'Micronesia, Fed. Sts.',
       'Antigua and Barbuda', 'Nepal', 'Lithuania', 'Vanuatu',
       'Afghanistan', 'Kuwait', 'Argentina', 'Panama', 'Oman', 'France',
       'Bosnia and Herzegovina', 'Mauritania', 'Somalia', 'Azerbaijan',
       'Maldives', 'Guinea-Bissau', 'Solomon Islands', 'Congo, Dem. Rep.',
       'Namibia', 'Eswatini', 'Nigeria', 'United Arab Emirates',
       'Burundi', 'Tajikistan', 'Honduras', 'Colombia', 'Iceland',
       'Morocco', 'Pakistan', 'Bolivia', 'Cambodia', 'Malaysia',
       'Dominican Republic', 'Italy', 'Vietnam', 'Albania', 'Czechia',
       'Tonga', 'Slovenia', 'Zambia', 'Egypt, Arab Rep.',
       'Papua New Guinea', 'Ireland', 'Chile', 'Syrian Arab Republic',
       'Serbia', 'Belgium', 'Cuba', 'Trinidad and Tobago', 'Botswana',
       'Paraguay', 'Malawi', 'Montenegro', 'Timor-Leste', 'Chad',
       'Sierra Leone', 'Mali', 'Bangladesh', 'Latvia', 'Angola',
       'Jamaica', 'China', 'Tanzania', 'Ecuador', 'Djibouti',
       "Cote d'Ivoire", 'Nicaragua', 'Iraq', 'Myanmar', 'Bahrain',
       'Cabo Verde', 'Uganda', 'St. Lucia', 'Belarus', 'Senegal',
       'Mongolia', 'Haiti', 'Niger', 'Slovak Republic', 'Tunisia',
       'Thailand', 'Samoa', 'Libya', 'Bulgaria', 'Netherlands', 'Liberia',
       'Ethiopia', 'Benin', 'New Zealand', 'Rwanda',
       'Sao Tome and Principe', 'Guatemala', 'Cyprus', 'Venezuela, RB',
       'Portugal', 'Equatorial Guinea', 'Iran, Islamic Rep.', 'Lao PDR',
       'Mexico', 'Lebanon', 'Turkmenistan', 'Indonesia', 'United States',
       'Peru', 'Mozambique', 'United Kingdom', 'Luxembourg', 'Sri Lanka',
       'Uzbekistan', 'Lesotho', 'Guinea', 'Poland', 'Canada', 'Suriname',
       'Mauritius', 'Barbados', 'El Salvador', 'Burkina Faso', 'Qatar',
       'Fiji', 'Australia', 'North Macedonia', 'Georgia']

socio_prof_selec = [0,1,2,3,4]
diet_habits_selec = [0,1,2,3,4]
phys_act_selec = [0,1,2,3,4]
drink_selec = [0,1,2,3,4]
smok_selec = [1,2,3,4,5]
fam_selec = [1,2,3,4,5]
chronic_selec = [0,1,2,3,4,5]


sex_selec = ['Female', 'Male', 'Other']

#Loading model
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, '..', 'model', 'model.pkl')
preprocessor_path = os.path.join(script_dir, '..', 'model', 'preprocessor.joblib')

url_model = model_path
url_preprocessor = preprocessor_path

model = joblib.load(url_model)
preprocessor = joblib.load(url_preprocessor)


def app():

    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("Enter your informations below")

    with col1:
        country = st.selectbox("Country", countries)
        soc_p = st.selectbox("Socio-professional category", socio_prof_selec)
        age = st.slider("Age", min_value=18, max_value=100)
        sex = st.selectbox("Sex", sex_selec)

    with col2:
        st.write("")
        st.write("")
        height = st.slider("Height", min_value=100, max_value=220, step=1)
        weight = st.slider("Weight (kg)", min_value=30, max_value=250, step=1)
        bmi = weight / ((height/100)**2)
        st.write(f"BMI {round(bmi,1)}")
        diet = st.selectbox("Diet Habits (from best to worse)", diet_habits_selec)
        phys = st.selectbox("Physical Activity (from best to worse)", phys_act_selec)

    with col3:
        st.write("")
        st.write("")
        drink = st.selectbox("Drinking habits (from best to worse)", drink_selec)
        smoke = st.selectbox("Smoking habits (from best to worse)", smok_selec)
        fam = st.selectbox("Family diseases history (from best to worse)", fam_selec)
        chronic = st.selectbox("Chronic diseases (from best to worse)", chronic_selec)


    dob = 2023 - age

    if sex == 'Male':
        sex_number = 1
    else:
        sex_number = 0

    #need to be modified to remove Family Stat, BMI model as well
    predict_data = [{'Country': country,
                     'DoB': dob,
                     'Sex': sex_number,
                     'Socpro': soc_p,
                     'BMI': bmi,
                     'Diet_Hab': diet,
                     'Phys_act': phys,
                     'Drink_stat': drink,
                     'Smoking_stat': smoke,
                     'Family_dis': fam,
                     'Chronic': chronic}]

    df = pd.DataFrame(predict_data)

    data_array = df

    data_array_proproc = preprocessor.transform(data_array)

    prediction = model.predict(data_array_proproc)

    with col2:
        st.write('')
        final_result_a = min(100, prediction[0]-2)
        final_result = max(final_result_a, age)
        st.write(f"Life Expectancy Prediction: {round(final_result,1)} years")



if __name__ == "__main__":
    app()
