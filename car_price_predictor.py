# For deployment
import streamlit as st
import numpy as np
import pickle
import pandas as pd

# use streamlit run app.py to run the app

# creating a basic flask server

from flask import Flask

app = Flask(__name__)

model = pickle.load(open('LR.pkl', 'rb'))
df = pd.read_csv('Car_Data.csv')


def clean_inputs(car_name, present_price, fuel_type, seller_type, transmission):
    encoded_car_name = model['Car_Name'][car_name]
    encoded_present_price = present_price / 100000
    encoded_fuel_type = model['Fuel_Type'][fuel_type]
    encoded_seller_type = model['Seller_Type'][seller_type]
    encoded_transmission = model['Transmission'][transmission]

    return encoded_car_name, encoded_present_price, encoded_fuel_type, encoded_seller_type, encoded_transmission


def normalize(x, mu, std):
    return (x - mu) / std


def main():
    st.title('Vehicle Price Predictor')
    car_name = st.selectbox('Select model of vehicle',
                            tuple(df['Car_Name'].sort_values().unique()))

    year = st.slider('Year of Purchase',
                     min_value=2000,
                     max_value=2021,
                     value=2000,
                     step=1)
    present_price = st.number_input('Enter Present Price in rupees',
                                    min_value=10000,
                                    max_value=10000000)
    kms_driven = st.number_input('Enter Kms driven',
                                 min_value=1,
                                 max_value=1000000)
    fuel_type = st.selectbox('Select fuel type',
                             tuple(df['Fuel_Type'].sort_values().unique()))

    seller_type = st.selectbox('Select Seller Type',
                               tuple(df['Seller_Type'].sort_values().unique()))

    transmission = st.selectbox(
        'Select Transmission', tuple(df['Transmission'].sort_values().unique()))

    owner = st.slider('Number of Owners',
                      min_value=0,
                      max_value=4,
                      value=0,
                      step=1)

    if st.button('Get Price'):
        car_name, present_price, fuel_type, seller_type, transmission = clean_inputs(
            car_name, present_price, fuel_type, seller_type, transmission)

        x = np.array([
            car_name, year, present_price, kms_driven, fuel_type, seller_type,
            transmission, owner
        ])

        x = normalize(x, model['mean'], model['std'])

        weights = model['weights']
        bias = model['bias']

        predicted_price = np.dot(x, weights) + bias
        st.header('Predicted Price: ₹' +
                  str(np.round(predicted_price * 100000, 2)))


if __name__ == '__main__':
    main()
