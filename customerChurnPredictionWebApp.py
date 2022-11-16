# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 23:42:31 2022

@author: Pc
"""

import pickle
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

         
app_mode = st.sidebar.selectbox('Select Page',['Home','Predict_customer_Churn'])
    
if app_mode=='Home': 
    st.title('Customer Data  Visualization')
    st.sidebar.info('This app is created for prediction of Customer Churn as well as the visualization of the customer information/data')

    
    file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])
    if file_upload is not None:
            data = pd.read_csv(file_upload)
            
    def data_summary():
        st.header('Statistics of Dataframe')
        st.write(data.describe())
                
    def data_header():
        st.header('Header of Dataframe')
        st.write(data.head())
                    
         # a scatter plot           
    def displayplot():
        st.header('Plot of Data')
    
        fig, ax = plt.subplots(1,1)
        ax.scatter(x=data['current_balance'], y=data['current_month_balance'])
        ax.set_xlabel('current_balance')
        ax.set_ylabel('current_month_balance')
    
        st.pyplot(fig)
    
    #incase they want an interactive plot
    def interactive_plot():
        col1, col2 = st.columns(2)
    
        x_axis_val = col1.selectbox('Select the X-axis', options=data.columns)
        y_axis_val = col2.selectbox('Select the Y-axis', options=data.columns)

        plot = px.scatter(data, x=x_axis_val, y=y_axis_val)
        st.plotly_chart(plot, use_container_width=True)
        
       
# Sidebar setup
    st.sidebar.title('Navigation')
    options = st.sidebar.radio('Select what you want to display:', [ 'Data Summary', 'Data Header', 'Scatter Plot', 'Fancy Plots'])

# Navigation options
    if options == 'Data Summary':
            data_summary()
    elif options == 'Data Header':
                data_header()
    elif options == 'Scatter Plot':
        displayplot()
    elif options == 'Fancy Plots':
        interactive_plot()
     
        
     
elif app_mode == 'Predict_customer_Churn':
    st.title("Predicting Customer Churn")

# input widgets to collect customer details and generate predictions,
# specify our inputs
    st.subheader('Fill in customer details to get prediction ')
    st.sidebar.header("Prediction with features used to train the Logistics Regression Model")
    current_month_debit = st.number_input("current_month_debit")
    occupation_retired = st.selectbox('Customer is retired:', ['yes', 'no'])
    occupation_salaried = st.selectbox('Customer is employed with a salary:', ['yes', 'no'])
    occupation_self_employed = st.selectbox('Customer is self employed:', ['yes', 'no'])
    occupation_student = st.selectbox('Customer is a student:', ['yes', 'no'])
    previous_month_debit = st.number_input("previous_month_debit")
    current_balance = st.number_input("current_balance")
    previous_month_end_balance = st.number_input("previous_month_end_balance")
    vintage = st.number_input("vintage")
    output= ""
    output_prob = ""
    
    # Pre-processing user input    
    if occupation_retired == "no":
        occupation_retired = 0
    else:
        occupation_retired = 1

    if occupation_salaried == "yes":
        occupation_salaried = 1
    else:
        occupation_salaried = 0
 
    if  occupation_self_employed == "yes":
        occupation_self_employed = 1
    else:
        occupation_self_employed = 0    
 
    if occupation_student == "yes":
        occupation_student = 1
    else:
        occupation_student = 0
    features = [current_month_debit, previous_month_debit, current_balance, previous_month_end_balance, vintage, occupation_retired, occupation_salaried, occupation_self_employed, occupation_student]
    results = np.array(features).reshape(1, -1)
    
    
    if st.button("Predict"):
        #load the saved model
        loaded_model = pickle.load(open('C:/Users/Pc/Desktop/My Project/customer_churn_prediction.pkl', 'rb'))
        
        prediction = loaded_model.predict_proba(results)[0, 1]
        churn = prediction >= 0.3
        output_prob = float(prediction)
        output = bool(churn)
    st.success('Churn: {0}, Risk Score: {1}'.format(output, output_prob))

        
   
      
     
   
    

   
