import streamlit as st
import joblib
import pandas as pd
import os

st.set_page_config(
    page_title="Life Prediction (based on really fake data)",
    layout='wide'
)

#list of countries
countries = sorted(['Turkiye', 'Spain', 'India', 'Guyana', 'Israel', 'Costa Rica',
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
       'Fiji', 'Australia', 'North Macedonia', 'Georgia'])

#numeric and string values for categories
socio_prof_selec = [0,1,2,3,4]
diet_habits_selec = [0,1,2,3,4]
phys_act_selec = [0,1,2,3,4]
drink_selec = [0,1,2,3,4]
smok_selec = [1,2,3,4,5]
fam_selec = [1,2,3,4,5]
chronic_selec = [0,1,2,3,4,5]

socpro_mapping = {
    0: 'Low-Income/Poverty',
    1: 'Lower-Middle-Income',
    2: 'Middle-Income',
    3: 'Upper-Middle-Income',
    4: 'High-Income'
}

diet_mapping = {
    4: 'Very unhealthy',
    3: 'Unhealthy',
    2: 'Moderate',
    1: 'Balanced',
    0: 'Healthy'
}

phys_act_mapping = {
    0: 'Very active',
    1: 'Active',
    2: 'Moderate',
    3: 'Inactive',
    4: 'Very inactive'
}

drink_mapping = {
    0: 'Abstainer',
    1: 'Rarely',
    2: 'Occasional',
    3: 'Regular',
    4: 'Frequent'
}

smoke_mapping = {
    1: 'Non-smoker',
    2: 'Occasional smoker',
    3: 'Regular smoker',
    4: 'Heavy smoker',
    5: 'Chain smoker'
}

chronic_mapping = {
    0: 'No chronic diseases',
    1: '1 chronic disease',
    2: '2 chronic diseases',
    3: '3 chronic diseases',
    4: '4 chronic diseases',
    5: 'More than 4'
}

sex_selec = ['Female', 'Male', 'Other']

#Importing preprocessor and model
script_dir = os.path.dirname(os.path.abspath(__file__))
preprocessor_path = os.path.join(script_dir, '..', 'model', 'preprocessor.joblib')
preprocessor = joblib.load(preprocessor_path)

model_path = os.path.join(script_dir, '..', 'model', 'model.pkl')
model = joblib.load(model_path)

def app():

    st.markdown(f"<h1 style='text-align: center;'><br><br>Enter your informations below</span></h1>", unsafe_allow_html=True)
    st.write("")
    st.write("")

    col1, col2, col3 = st.columns(3)

    st.write("")
    st.write("")
    st.write('None of those predictions are reliable and it should not be taken seriously.')

    with col1:
        country = st.selectbox("Country", countries)
        soc_p = st.selectbox("Socio-professional category", [socpro_mapping[val] for val in socio_prof_selec])
        socpro_numeric = [key for key, value in socpro_mapping.items() if value == soc_p][0]
        sex = st.selectbox("Sex", sex_selec)
        height = st.slider("Height", min_value=100, max_value=220, step=1)
        weight = st.slider("Weight (kg)", min_value=30, max_value=250, step=1)
        bmi = weight / ((height/100)**2)

        if bmi < 16 or bmi >= 40:
            color_bmi = 'red'
        elif 30 <= bmi < 40 or 16 <= bmi < 18:
            color_bmi = 'orange'
        elif 25 <= bmi < 30:
            color_bmi = 'yellowgreen'
        else:
            color_bmi = 'green'

        st.markdown(f"<h5><span style='color: {color_bmi};'>BMI : {round(bmi,1)}</span></h5>",unsafe_allow_html=True)

    with col2:
        diet = st.selectbox("Diet Habits", [diet_mapping[val] for val in diet_habits_selec])
        phys = st.selectbox("Physical Activity", [phys_act_mapping[val] for val in phys_act_selec])
        drink = st.selectbox("Drinking habits", [drink_mapping[val] for val in drink_selec])
        smoke = st.selectbox("Smoking habits", [smoke_mapping[val] for val in smok_selec])
        chronic = st.selectbox("Chronic diseases", [chronic_mapping[val] for val in chronic_selec])
        diet_numeric = [key for key, value in diet_mapping.items() if value == diet][0]
        phys_numeric = [key for key, value in phys_act_mapping.items() if value == phys][0]
        drink_numeric = [key for key, value in drink_mapping.items() if value == drink][0]
        smoke_numeric = [key for key, value in smoke_mapping.items() if value == smoke][0]
        chronic_numeric = [key for key, value in chronic_mapping.items() if value == chronic][0]

    if sex == 'Male':
        sex_number = 1
    else:
        sex_number = 0

    predict_data = [{'Country': country,
                     'Sex': sex_number,
                     'Socpro': socpro_numeric,
                     'BMI': bmi,
                     'Diet_Hab': diet_numeric,
                     'Phys_act': phys_numeric,
                     'Drink_stat': drink_numeric,
                     'Smoking_stat': smoke_numeric,
                     'Chronic': chronic_numeric}]

    df = pd.DataFrame(predict_data)

    data_array = df

    data_array_proproc = preprocessor.transform(data_array)

    prediction = model.predict(data_array_proproc)

    with col3:
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        final_result_a = min(100, prediction[0]-2)
        final_result = max(final_result_a, 0)

        if final_result < 40:
            color = 'red'
        elif 40 <= final_result < 60:
            color = 'orange'
        elif 60 <= final_result < 80:
            color = 'yellowgreen'
        else:
            color = 'green'

        st.markdown(f"<h2 style='text-align: center;'><br><br>Life Expectancy Prediction:<br> <span style='color: {color};'>{round(final_result, 1)} years</span></h2>", unsafe_allow_html=True)



if __name__ == '__main__':
    app()
