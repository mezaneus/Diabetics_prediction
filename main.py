import streamlit as st ##### UI inteface is our ML model
import pandas as pd #### is used for accessing dataframe logics
from sklearn.model_selection import train_test_split #### To load ML model and use it 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv('diabetes2.csv')

#### To predict the outcome , whether the patient with BMI,pregencies,glucose,BP and insulin values ...is having diabetes or not######
y = df['Outcome'] #### output ####
X = df.drop('Outcome', axis = 1) #### axis = 0 means rows and axis = 1 means column #### Input #####

#### Training and Testing Dataset ######

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)

### Classification algorithm ######
model = LogisticRegression()
model.fit(X_train,y_train)

####Predicting function ####

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)

##### UI Interface ####

st.title("Diabetes Prediction")

###user inputs

Pregnancies = st.number_input("Pregnancies")
Glucose = st.number_input("Glucose")
BloodPressure = st.number_input("BloodPressure")
SkinThickness = st.number_input("SkinThickness")
Insulin = st.number_input("Insulin")
BMI = st.number_input("BMI")
DiabetesPedigreeFunction = st.number_input("DiabetesPedigreeFunction")
Age = st.number_input("Age")

input_data = {
    "Pregnancies":Pregnancies,
    "Glucose": Glucose,
    "BloodPressure" : BloodPressure,
    "SkinThickness": SkinThickness,
    "Insulin" : Insulin,
    "BMI" : BMI,
    "DiabetesPedigreeFunction" : DiabetesPedigreeFunction,
    "Age": Age  

}

####Convert to dataframe

input_df = pd.DataFrame([input_data])

### predict button
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    if prediction == 1:
        st.write("You have Diabetes")
    else :
        st.write("You are Healthy")

#######Display
st.write("Sample")
st.dataframe(df)