# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 20:14:33 2023

@author: Bamise
"""

import numpy as np
import pickle
import streamlit as st
import scikit-learn as sklearn

with open('model_diabetesprediction.pkl', 'rb') as file:
    loaded_model = pickle.load(file)
    
    
#create a function for prediction

def diabetes_prediction(input_data):
    
    #convert input data to float
    input_data_as_float = [float(value) for value in input_data]
    
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data_as_float)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
      return 'The person is not diabetic'
    else:
      return 'The person is diabetic'
  
    
def main():
    
    #Title
    st.title('Live Diabetes Prediction Web App')
    
    # get input data from user
    pregnancies = st.text_input("Number of Pregnancies")
    
    glucose = st.text_input("Glucose level (mg/dl)")
    
    bmi =  st.text_input("BMI value")
    
    age = st.text_input("Age (years)")
    
    #Code for prediction
    diagnosis = ''
    
    # create a button for prediction
    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction([pregnancies, glucose, bmi, age])
    
    st.success(diagnosis)
    
if __name__ == '__main__':
    main()
