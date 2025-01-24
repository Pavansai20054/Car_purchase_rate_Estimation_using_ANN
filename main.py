import streamlit as st
import pandas as pd
from tensorflow.keras.models import load_model
import pickle
from sklearn.preprocessing import LabelEncoder

# Streamlit page config
st.set_page_config(
    page_title="Car Purchase Rate Estimation",
    page_icon="💵🚗"
)

st.title("Estimation of Amount Paid for a Car")

# Load and compile the model
model = load_model("car_purchase_rate.h5") 

# Load the fitted LabelEncoder from the saved file
with open('label_encoder.pkl', 'rb') as file:
    le = pickle.load(file)

# Get the list of countries from the LabelEncoder's classes_ attribute
countries = le.classes_

col1, col2 = st.columns(2)

with col1:

    # User inputs
    country = st.selectbox("Select your Country:", countries)  # Display countries in selectbox
    annual_salary = st.number_input("Enter your salary:", min_value=0.0, max_value=10000000.0, value=0.0, step=0.1)
    credit_card_debt = st.number_input("Enter your Credit card debt:", min_value=0.0, max_value=1000000.0, value=0.0, step=0.1)
    networth = st.number_input("Enter your net worth:", min_value=0.0, max_value=1000000.0, value=0.0, step=0.1)

    # Display message that model is loaded
    st.write("Model successfully loaded and compiled.")

    # Button to make the prediction
    if st.button("Estimate Car Purchase Rate"):
        # Prepare the input data for the model
        input_data = pd.DataFrame({
            'country': [country],
            'annual_salary': [annual_salary],
            'credit_card_debt': [credit_card_debt],
            'networth': [networth]
        })

        # Encode the country input to integer using LabelEncoder
        input_data['country'] = le.transform([country])[0]  # Transform the country input to an integer label

        # Use the model to predict the car purchase rate
        prediction = model.predict(input_data)

        # Scale down the prediction by a factor (for example, divide by 100,000 to make it more manageable)
        scaled_prediction = prediction[0].item() / 100000  # Adjust the scaling factor as needed

        # Display the formatted prediction value with commas and 4 decimal places
        st.write(f"The estimated car purchase rate is: ${scaled_prediction:,.4f}")

with col2:
    st.image("image\image.jpg")