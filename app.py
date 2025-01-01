import pandas as pd
import numpy as np
import pickle as pk
import streamlit as st

# Load your pre-trained model
model = pk.load(open('house_model (1).pkl', 'rb'))

# App Title and Description
st.set_page_config(page_title="House Price Prediction", page_icon="🏠")
st.title("House Price Prediction App 🏠")
st.markdown("""
This app predicts the **estimated price** of a house based on the details you provide.
Fill in the fields below to get started!
""")

# Example dataset for user input reference
data = pd.read_csv('99clean_master_filtered.csv')

# Input fields for housing details
district = st.selectbox('District', data['district'].unique())
city = st.selectbox('City', data['city'].unique())
bedrooms = st.number_input('Number of Bedrooms', min_value=1, max_value=10, value=2, step=1)
bathrooms = st.number_input('Number of Bathrooms', min_value=1, max_value=10, value=2, step=1)
land = st.number_input('Land Area (m²)', min_value=20, max_value=1000, value=100, step=1)
floor = st.number_input('Floor Area (m²)', min_value=20, max_value=1000, value=80, step=1)

# Predict button
if st.button('Predict'):
    # Create a DataFrame for the user inputs
    house_details = pd.DataFrame([[district, city, bedrooms, bathrooms, land, floor]],
                                 columns=['district', 'city', 'bedrooms', 'bathrooms', 'land', 'floor'])
    st.write('House Details:', house_details)
    
    # Predict the house price using the model
    house_price = model.predict(house_details)[0]

    # Format the predicted price
    formatted_price = 'Rp {:,.0f}'.format(house_price).replace(',', '.')
    st.markdown(f'Estimated Price: {formatted_price}')

# Footer
st.markdown("---")
st.markdown("© 2024 - House Price Prediction App by Jonathan")

# Run the app.py
# streamlit run app.py
