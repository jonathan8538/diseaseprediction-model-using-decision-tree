import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# Load preprocessed dataset
@st.cache_data
def load_data():
    file_path = r'Data\cleaned_data.csv'
    df = pd.read_csv(
        file_path,
        encoding='latin1',
        header=None,
        on_bad_lines='skip'  # Skip problematic lines
    )
    df.columns = ['Disease', 'Symptom', 'Count']
    return df

# Load data
df = load_data()

# Encode categorical features
le_symptom = LabelEncoder()
le_disease = LabelEncoder()

X = le_symptom.fit_transform(df['Symptom']).reshape(-1, 1)
y = le_disease.fit_transform(df['Disease'])

# Train the model
clf_dt = DecisionTreeClassifier()
clf_dt.fit(X, y)

# Streamlit UI
st.title("HealthyDiseases: Symptom-Based Disease Prediction")
st.write("### Welcome!")
st.write(
    """
    This tool predicts potential diseases based on symptoms you provide. You can select one or more symptoms from the sidebar menu to predict the disease.</small>
    """,
    unsafe_allow_html=True,
)

# Add a section explaining the algorithm
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

# Sidebar for input symptoms
st.sidebar.header("Input Symptoms")

# Filter out NaN values and the "symptom" entry
symptom_options = [
    symptom for symptom in df['Symptom'].dropna().unique() 
    if isinstance(symptom, str) and symptom.lower() != "symptom"
]

# Multiselect for selecting one or more symptoms
selected_symptoms = st.sidebar.multiselect("Select One or More Symptoms", symptom_options)


# Predict the disease
if st.sidebar.button("Predict"):
    if selected_symptoms:
        input_symptoms_encoded = le_symptom.transform(selected_symptoms).reshape(-1, 1)
        predictions_encoded = clf_dt.predict(input_symptoms_encoded)
        predictions = le_disease.inverse_transform(predictions_encoded)
        
        # Display results in a single line and larger font
        st.markdown(
            f"<h3 style='color: #4CAF50;'>Predicted Diseases: {', '.join(predictions)}</h3>",
            unsafe_allow_html=True,
        )
    else:
        st.write("Please select at least one symptom.")
