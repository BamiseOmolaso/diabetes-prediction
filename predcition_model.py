# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# Ignore all warnings
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pickle 

with open('C:/Users/user/OneDrive/Documents/Everything AI/model_diabetesprediction.pkl', 'rb') as file:
    loaded_model = pickle.load(file)
    
input_data = (6,70,28.8,51)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')