import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(layout="wide")
st.title("Iris Prediction app")


# reading all the pickle files
model_scaler= pickle.load(open('scaler_iris.pkl','rb')) # 1st pickle scale model
model_lr_basic= pickle.load(open('model_lr_basic_iris.pkl','rb')) # 2nd pickle scale model
model_lr_smote= pickle.load(open('model_lr_smote_iris.pkl','rb')) # 3rd pickle scale model



# users need to define the input
st.header("Enter the input values by User")

sepal_length= st.number_input("Enter the integer value for sepal length(cm)")
sepal_width= st.number_input("Enter the integer value for sepal width(cm)")
petal_length= st.number_input("Enter the integer value for petal length(cm)")
petal_width= st.number_input("Enter the integer value for petal width(cm)")



#create a dictionary for user_input
user_input={'sepal length (cm)':sepal_length,
            'sepal width (cm)':sepal_width,
            'petal length (cm)':petal_length,
            'petal width (cm)':petal_width}

#convert to DataFrame
user_input_df= pd.DataFrame(user_input, index=[0])

#scale the user_data
user_input_df_scaled=model_scaler.transform(user_input_df)

st.write("Basic Model is simple logistic regression model using default parameters")

#user will select the model
selected_model=st.selectbox("Select one of the following models",("Basic Model", "Smote Model"))
if st.button("Predict"):
      
   if selected_model=="Basic Model":
       prediction=model_lr_basic.predict(user_input_df_scaled)
       st.write("Basic Model is simple logistic regression model using default parameters")

   elif selected_model== "Smote Model":
       prediction=model_lr_smote.predict(user_input_df_scaled)


   result=prediction[0]
   if result==0:
       st.success("This is Iris")
   else:
       st.success("This is not Iris")
