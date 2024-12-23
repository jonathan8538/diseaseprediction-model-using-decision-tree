import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder


@st.cache_data
def load_data():
    file_path = 'Data/cleaned_data.csv'
    df = pd.read_csv(
        file_path,
        encoding='latin1',
        header=None,
        on_bad_lines='skip'
    )
    df.columns = ['Disease', 'Symptom', 'Count']
    return df


df = load_data()


le_symptom = LabelEncoder()
le_disease = LabelEncoder()

X = le_symptom.fit_transform(df['Symptom']).reshape(-1, 1)
y = le_disease.fit_transform(df['Disease'])


clf_dt = DecisionTreeClassifier()
clf_dt.fit(X, y)


st.title("HealthyDiseases: Symptom-Based Disease Prediction")
st.write("### Welcome!")
st.write(
    """
    This tool predicts potential diseases based on symptoms you provide. You can select one or more symptoms from the sidebar menu to predict the disease.</small>
    """,
    unsafe_allow_html=True,
)

st.write("#### How It Works:")
st.write(
    """
    - **Data Preparation**: The data is preprocessed to clean missing values and align symptoms with diseases.
    - **Encoding**: Symptoms and diseases are converted into numerical labels using `LabelEncoder`.
    - **Model Training**: A Decision Tree Classifier learns patterns between symptoms and diseases.
    - **Prediction**: Based on selected symptoms, the model predicts the most likely disease.
    """,
    unsafe_allow_html=True,
)


st.sidebar.header("Input Symptoms")


symptom_options = [
    symptom for symptom in df['Symptom'].dropna().unique() 
    if isinstance(symptom, str) and symptom.lower() != "symptom"
]


selected_symptoms = st.sidebar.multiselect("Select One or More Symptoms", symptom_options)



if st.sidebar.button("Predict"):
    if selected_symptoms:
        input_symptoms_encoded = le_symptom.transform(selected_symptoms).reshape(-1, 1)
        predictions_encoded = clf_dt.predict(input_symptoms_encoded)
        predictions = le_disease.inverse_transform(predictions_encoded)
        
        
        st.markdown(
            f"<h3 style='color: #4CAF50;'>Predicted Diseases: {', '.join(predictions)}</h3>",
            unsafe_allow_html=True,
        )
    else:
        st.write("Please select at least one symptom.")
