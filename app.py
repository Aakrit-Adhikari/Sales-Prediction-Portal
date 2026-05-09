import streamlit as st 
import joblib
import numpy as np


scaler=joblib.load("scaler.pkl")
model=joblib.load("model.pkl")

st.title("Customer car price estimator app")
st.divider()
st.write("""This app is getting a price stimation for the customer so a car with the price range given can be adviced to the customer""")

age= st.number_input("Enter the age",min_value=18,max_value=90,value=40,step=1)
salary=st.number_input("Enter salary", min_value=1000,max_value=999999999, value=30000,step=5000)
networth=st.number_input("Enter networth",min_value=0,max_value=999999,value=100000)

X=[age,salary,networth]
calculatebutton=st.button("Calculate")

st.divider()

if calculatebutton:
  
    st.balloons()
    X_2=np.array(X)
    X_array=scaler.transform([X_2])
    prediction=model.predict(X_array)
    st.write(f"Predictions is {prediction[0]:,.2f}")
    st.write("Advice car in the similar values")

else:
    st.write("Please enter value and press enter butn")

