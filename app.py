import os
import joblib
import pandas as pd
import streamlit as st
import anthropic
from sklearn.preprocessing import RobustScaler
from dotenv import load_dotenv

load_dotenv()

# Load the model and scaler
model = joblib.load('feature_selected_rf_model.joblib')
scaler = joblib.load('robust_scaler.joblib')

anthropic.api_key = os.getenv("ANTHROPIC_API_KEY")

def preprocess_features(features):
    # Define the correct order of features
    feature_order = ['GenHlth', 'BMI', 'HighBP', 'Age', 'Income', 'HighChol', 
                     'Education', 'HvyAlcoholConsump', 'PhysActivity', 'DiffWalk']
    
    # Create a DataFrame with features in the correct order
    df = pd.DataFrame([features], columns=feature_order)
    
    # Scale only the 'BMI' feature
    df['BMI'] = scaler.transform(df[['BMI']])

    binary_features = ['HighBP', 'HighChol', 'PhysActivity',
                       'HvyAlcoholConsump', 'DiffWalk']

    for feature in binary_features:
        df[feature] = df[feature].astype(int)

    print(df.values.tolist())
    
    # Convert back to a list of lists
    return df.values.tolist()

def make_prediction(preprocessed_features):
    prediction = model.predict(preprocessed_features)
    print(prediction)
    return prediction[0]

def get_claude_interpretation(features, prediction):
    prompt = f"""
    Given the following input features for a patient:
    - General Health: {["Excellent", "Very Good", "Good", "Fair", "Poor"][features['GenHlth']-1]}
    - BMI: {features['BMI']}
    - High Blood Pressure: {"Yes" if features['HighBP'] == 1 else "No"}
    - Age Category: {features['Age']}
    - Income Level: {features['Income']}
    - High Cholesterol: {"Yes" if features['HighChol'] == 1 else "No"}
    - Education Level: {features['Education']}
    - Heavy Alcohol Consumption: {"Yes" if features['HvyAlcoholConsump'] == 1 else "No"}
    - Physical Activity: {"Yes" if features['PhysActivity'] == 1 else "No"}
    - Difficulty Walking: {"Yes" if features['DiffWalk'] == 1 else "No"}

    The meaning of the features and their values are given below:

    HighBP

    0 = no high BP 1 = high BP

    HighChol

    0 = no high cholesterol 1 = high cholesterol

    BMI

    Body Mass Index value scaled by RobustScaler

    PhysActivity

    physical activity in past 30 days - not including job 0 = no 1 = yes

    HvyAlcoholConsump

    (adult men >=14 drinks per week and adult women>=7 drinks per week) 0 = no 1 = yes

    GenHlth

    Would you say that in general your health is: scale 1-5 1 = excellent 2 = very good 3 = good 4 = fair 5 = poor

    DiffWalk

    Do you have serious difficulty walking or climbing stairs? 0 = no 1 = yes

    Age

    13-level age category:
    1: "18-24", 2: "25-29", 3: "30-34", 4: "35-39", 5: "40-44", 
    6: "45-49", 7: "50-54", 8: "55-59", 9: "60-64", 10: "65-69", 
    11: "70-74", 12: "75-79", 13: "80 or older

    Education

    Education level scale 1-6:
    1: "Never attended school or only kindergarten", 
    2: "Elementary", 
    3: "Some high school", 
    4: "High school graduate", 
    5: "Some college or technical school", 
    6: "College graduate"

    Income

    Income scale: scale 1-8
    1: "Less than $10,000", 
    2: "Less than $15,000", 
    3: "Less than $20,000", 
    4: "Less than $25,000", 
    5: "Less than $35,000", 
    6: "Less than $50,000", 
    7: "Less than $75,000", 
    8: "$75,000 or more"
    
    The model predicted a {"high" if prediction == 1 else "low"} possibility of Diabetes.
    
    Please provide a brief interpretation of this prediction, explaining what it might mean for the patient's diabetic health. 
    Consider the impact of the various risk factors and suggest any lifestyle changes that might be beneficial.
    """
    
    response = anthropic.Anthropic().messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    
    return response.content[0].text

def main():
    st.title("Diabetes Prediction with LLM Interpretation")

    st.write("Please enter your information:")

    GenHlth = st.select_slider("In general, how would you say your health is?", 
                               options=[1, 2, 3, 4, 5],
                               format_func=lambda x: {1: "Excellent", 2: "Very Good", 3: "Good", 4: "Fair", 5: "Poor"}[x])
    BMI = st.number_input("Body Mass Index (BMI)", min_value=0.0, max_value=100.0, step=0.1)
    HighBP = st.radio("Do you have high blood pressure?", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    Age = st.selectbox("Age Category",
                       options=range(1, 14),
                       format_func=lambda x: {1: "18-24", 2: "25-29", 3: "30-34", 4: "35-39", 5: "40-44", 
                                              6: "45-49", 7: "50-54", 8: "55-59", 9: "60-64", 10: "65-69", 
                                              11: "70-74", 12: "75-79", 13: "80 or older"}[x])
    Income = st.selectbox("Income Level", 
                          options=range(1, 9),
                          format_func=lambda x: {1: "Less than $10,000", 
                                                 2: "Less than $15,000", 
                                                 3: "Less than $20,000", 
                                                 4: "Less than $25,000", 
                                                 5: "Less than $35,000", 
                                                 6: "Less than $50,000", 
                                                 7: "Less than $75,000", 
                                                 8: "$75,000 or more"}[x])
    HighChol = st.radio("Do you have high cholesterol?", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    Education = st.selectbox("Education Level", 
                             options=range(1, 7),
                             format_func=lambda x: {1: "Never attended school or only kindergarten", 
                                                    2: "Elementary", 
                                                    3: "Some high school", 
                                                    4: "High school graduate", 
                                                    5: "Some college or technical school", 
                                                    6: "College graduate"}[x])
    HvyAlcoholConsump = st.radio("Heavy Alcohol Consumption (adult men >=14 drinks per week and adult women>=7 drinks per week)", 
                                 options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    PhysActivity = st.radio("Have you engaged in physical activity in the past 30 days (not including job)?", 
                            options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    DiffWalk = st.radio("Do you have serious difficulty walking or climbing stairs?", 
                        options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

    # Create a dictionary of features in the correct order
    features = {
        'GenHlth': GenHlth,
        'BMI': BMI,
        'HighBP': HighBP,
        'Age': Age,
        'Income': Income,
        'HighChol': HighChol,
        'Education': Education,
        'HvyAlcoholConsump': HvyAlcoholConsump,
        'PhysActivity': PhysActivity,
        'DiffWalk': DiffWalk
    }

    if st.button("Predict Diabetes Likelihood"):
        # Preprocess the features
        preprocessed_features = preprocess_features(features)
        
        # Make prediction
        prediction = make_prediction(preprocessed_features)
        
        # Get Claude's interpretation
        interpretation = get_claude_interpretation(features, prediction)
        
        # Display results
        st.subheader("Prediction:")
        if prediction == 1:
            st.markdown("Possibility of Having Diabetes: <span style='color:red; font-weight:bold;'>High</span>", unsafe_allow_html=True)
        else:
            st.markdown("Possibility of Having Diabetes: <span style='color:green; font-weight:bold;'>Low</span>", unsafe_allow_html=True)
        
        st.subheader("LLM's Interpretation:")
        st.write(interpretation)

if __name__ == "__main__":
    main()