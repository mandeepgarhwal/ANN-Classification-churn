import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from tensorflow.keras.callbacks import TensorBoard
import pickle
from tensorflow.keras.models import load_model

model1 = load_model('churn_model.h5')

with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)
with open('one_hot_encoder_geography.pkl', 'rb') as f:
    one_hot_encoder_geography = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

st.title("Customer Churn Prediction")

geography = st.selectbox('Select Geography', one_hot_encoder_geography.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider("Age", 18,92)
balance = st.number_input('Balance')
credit_score = st.number_input('credit_score')
estimated_salary = st.number_input("Estimated Salary")
tenure = st.slider("Tenure", 0, 10)
num_of_products = st.slider("Number of products", 1, 5)
has_cr_card = st.selectbox("has credit card",[0,1] )
is_active_member = st.selectbox("is active member", [0, 1])

input_data = {
    'CreditScore': credit_score,
    'Gender': label_encoder_gender.transform([gender])[0],
    'Age': age,
    'Tenure': tenure,
    'Balance': balance,
    'NumOfProducts': num_of_products,
    'HasCrCard': has_cr_card,
    'IsActiveMember': is_active_member,
    'EstimatedSalary': estimated_salary
}
label_encoded_data = one_hot_encoder_geography.transform([[geography]]).toarray()
data_df = pd.DataFrame(label_encoded_data, columns=one_hot_encoder_geography.get_feature_names_out(['Geography']))
input_data = pd.DataFrame([input_data])
input_data = pd.concat([input_data.reset_index(drop=True), data_df], axis=1)
print(input_data)
input_data_scaled = scaler.transform(input_data)

prediction = model1.predict(input_data_scaled)
prediction_probability = prediction[0][0]

st.write("Churn probability is: ", prediction_probability)

if prediction_probability > 0.5:
    st.write("Customer is likely to churn")
else:
    st.write("Customer is not likely to churn")