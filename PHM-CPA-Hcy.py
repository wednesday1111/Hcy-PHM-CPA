# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 11:02:43 2022

@author: Kevin Boss
"""
from PIL import Image
#from streamlit_shap import st_shap
import streamlit as st
import pandas as pd 
import time
import plotly.express as px 
import seaborn as sns
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import mean_squared_error,confusion_matrix,accuracy_score,recall_score,precision_score,classification_report,roc_auc_score
import shap
#import catboost
#from catboost import CatBoostClassifier
import pickle
import xgboost as xgb
from xgboost import XGBClassifier
# load the saved model
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import numpy as np 

def xgb_shap_transform_scale(original_shap_values, Y_pred, which):    
    from scipy.special import expit    
    #Compute the transformed base value, which consists in applying the logit function to the base value    
    from scipy.special import expit 
    #Importing the logit function for the base value transformation    
    untransformed_base_value = original_shap_values.base_values[-1]    
    #Computing the original_explanation_distance to construct the distance_coefficient later on    
    original_explanation_distance = np.sum(original_shap_values.values, axis=1)[which]    
    base_value = expit(untransformed_base_value) 
    # = 1 / (1+ np.exp(-untransformed_base_value))    
    #Computing the distance between the model_prediction and the transformed base_value    
    distance_to_explain = Y_pred[which] - base_value    
    #The distance_coefficient is the ratio between both distances which will be used later on    
    distance_coefficient = original_explanation_distance / distance_to_explain    
    #Transforming the original shapley values to the new scale    
    shap_values_transformed = original_shap_values / distance_coefficient    
    #Finally resetting the base_value as it does not need to be transformed    
    shap_values_transformed.base_values = base_value    
    shap_values_transformed.data = original_shap_values.data    
    #Now returning the transformed array    
    return shap_values_transformed


plt.style.use('default')

st.set_page_config(
    page_title = 'Real-Time Fraud Detection',
    page_icon = '🕵️‍♀️',
    layout = 'wide'
)

# dashboard title
#st.title("Real-Time Fraud Detection Dashboard")
#st.markdown("<h1 style='text-align: center; color: black;'>机器学习： 实时识别出虚假销售</h1>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; color: black;'>Prediction of in-Hospital Mortality</h1>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; color: black;'>for Cirrhotic Patients with AKI (Hcy-PHM-CPA)</h1>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; color: black;'> </h1>", unsafe_allow_html=True)


# side-bar 
def user_input_features():
    st.sidebar.header('Make a prediction')
    st.sidebar.write('User input parameters below ⬇️')
    a0 = st.sidebar.slider('Hcy (μmol/L)', 0.0, 100.0, 0.0)
    a1 = st.sidebar.slider('BUN (mmol/L)', 1.0, 100.0, 0.0)
    a2 = st.sidebar.slider('Total cholesterol (mmol/L)', 0.5, 10.0, 0.0)
    a3 = st.sidebar.slider('Total bilirubin (μmol/L)', 0.0, 300.0, 0.0)
    a4 = st.sidebar.slider('Total protein (g/L)', 20.0, 100.0, 0.0)
    a5 = st.sidebar.slider('Cholinesterase (U/L)', 1000.0, 13000.0, 0.0)
    a6 = st.sidebar.slider('HDL (mmol/L)', 0.0, 3.0, 0.0)
    a7 = st.sidebar.slider('Bile acid (μmol/L)', 1.0, 100.0, 0.0)
    a8 = st.sidebar.slider('INR', 0.0, 6.0, 0.0)
    a9 = st.sidebar.slider('TT (second)', 14.0, 50.0, 0.0)
    a10 = st.sidebar.slider('RBC (10E12/L)', 1.0, 10.0, 0.0)
    a11 = st.sidebar.slider('MCHC (g/L)', 200.0, 400.0, 0.0)
    a12 = st.sidebar.slider('RDW (%)', 10.0, 30.0, 0.0)
    a13 = st.sidebar.slider('HCO3 (mmol/L)', 4.0, 40.0, 0.0)
    a14 = st.sidebar.selectbox("HE", (0, 1))
    a15 = st.sidebar.slider('HCT (%)', 10.0, 60.0, 0.0)

    
    output = [a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15]
    return output

outputdf = user_input_features()



with open('model20230620_apply.pkl', 'rb') as f:
    catmodel = pickle.load(f)

#st.header('👉 Make predictions in real time')
colnames = ['Hcy (μmol/L)',
    'BUN (mmol/L)','Total cholesteral (mmol/L)','Total bilirubin (μmol/L)','Total protein (g/L)',
                     'Cholinesterase (U/L)','HDL (mmol/L)','Bile acid (μmol/L)','INR','TT (second)','RBC (10E12/L)',
                     'MCHC (g/L)','RDW (%)','HCO3 (mmol/L)','HE','HCT (%)']
outputdf = pd.DataFrame([outputdf], columns= colnames)

#st.write('User input parameters below ⬇️')
#st.write(outputdf)


#p1 = catmodel.predict(dtest)[0]
#p2 = catmodel.predict_proba(dtest)
dtest = xgb.DMatrix(outputdf.iloc[:,1:])
p2 = catmodel.predict(dtest)
p2 = p2+(outputdf.iat[0,0]-0.1)/(85.8-0.1)
p2 = (p2-0.004)/(1.6-0.004)
p1 = 0
if p2 < 0.15:
    p1 = 'Low risk of death'
elif p2 >= 0.45:
    p1 = 'High risk of death'
else:
    p1 = 'Medium risk of death'
#p1 = (p2 >= 0.232816)*1#best threshold

#modify output dataframe
outputdf_1 = outputdf.iloc[:,0:7]
outputdf_2 = outputdf.iloc[:,7:16]

placeholder6 = st.empty()
with placeholder6.container():
    st.subheader('Part1: User input parameters below ⬇️')
    st.write(outputdf_1)
    st.write(outputdf_2)


placeholder7 = st.empty()
with placeholder7.container():
        st.subheader('Part2: Output results ⬇️')
        st.write(f'1. Predicted class: {p1}')
        st.write(f'2. Predicted class probability: {p2}')
   


