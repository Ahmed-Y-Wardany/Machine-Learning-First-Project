import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
import pickle

# Set page configuration
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

# Title and description
st.title("🎓 Student Performance Predictor")
st.markdown("""
This application predicts student scores based on their demographic and academic information.
Please select a model and fill out the form below to see the results.
""")

# Define model paths (from the second code)
MODEL_DIR = r"C:/Users/ahmed\Desktop/GitHub/Machine-Learning-First-Project/Models/"
MODEL_PATHS = {
    'MathScore': os.path.join(MODEL_DIR, "linearRegressionForMath.pkl"),
    'ReadingScore': os.path.join(MODEL_DIR, "linearRegressionForReading.pkl"),
    'WritingScore': os.path.join(MODEL_DIR, "linearRegressionForWriting.pkl")
}

# Model selection
model_option = st.sidebar.selectbox(
    "Select Prediction Model",
    ["Writing Score Prediction", "Math Score Prediction", "Reading Score Prediction"]
)

# Map model selection to target variable
model_mapping = {
    "Writing Score Prediction": "WritingScore",
    "Math Score Prediction": "MathScore", 
    "Reading Score Prediction": "ReadingScore"
}

target_variable = model_mapping[model_option]

# Define encoding mappings (from the second code)
encoding_mappings = {
    'Gender': {'male': 1, 'female': 0},
    'EthnicGroup': {'group A': 0, 'group B': 1, 'group C': 2, 'group D': 3, 'group E': 4},
    'ParentEduc': {
        "master's degree": 0, 
        "bachelor's degree": 1, 
        "associate's degree": 3,
        "some college": 5, 
        "high school": 4, 
        "some high school": 2
    },
    'LunchType': {'standard': 1, 'free/reduced': 0},
    'TestPrep': {'completed': 1, 'none': 0},
    'ParentMaritalStatus': {'married': 0, 'single': 1, 'divorced': 2, 'widowed': 3},
    'PracticeSport': {'never': 0, 'sometimes': 2, 'regularly': 1},
    'IsFirstChild': {'yes': 1, 'no': 0},
    'TransportMeans': {'private': 1, 'school_bus': 0, 'public': 2},
    'WklyStudyHours': {'> 10': 0, '5 - 10': 2, '< 5': 1}
}

# Define the expected feature order for each model
expected_features = {
    'MathScore': [
        'Gender', 'EthnicGroup', 'ParentEduc', 'LunchType', 'TestPrep',
        'ParentMaritalStatus', 'PracticeSport', 'IsFirstChild', 'TransportMeans',
        'WklyStudyHours', 'ReadingScore', 'WritingScore'
    ],
    'ReadingScore': [
        'Gender', 'EthnicGroup', 'ParentEduc', 'LunchType', 'TestPrep',
        'ParentMaritalStatus', 'PracticeSport', 'IsFirstChild', 'TransportMeans',
        'WklyStudyHours', 'MathScore', 'WritingScore'
    ],
    'WritingScore': [
        'Gender', 'EthnicGroup', 'ParentEduc', 'LunchType', 'TestPrep',
        'ParentMaritalStatus', 'PracticeSport', 'IsFirstChild', 'TransportMeans',
        'WklyStudyHours', 'MathScore', 'ReadingScore'
    ]
}

# Load the selected model
try:
    model_path = MODEL_PATHS[target_variable]
    
    # Try to load with joblib first, then with pickle if that fails
    try:
        model = joblib.load(model_path)
    except:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
            
    st.sidebar.success(f"{model_option} model loaded successfully!")
except Exception as e:
    st.sidebar.error(f"Error loading model: {e}")
    st.sidebar.info(f"Looking for model at: {MODEL_PATHS[target_variable]}")
    st.stop()

def preprocess_input(input_dict, target_variable):
    """Preprocess the input data to match the model's training format"""
    processed = input_dict.copy()
    
    # Encode categorical variables
    for feature, mapping in encoding_mappings.items():
        if feature in processed:
            processed[feature] = mapping[processed[feature]]
    
    # Convert to DataFrame with the correct feature order
    processed_df = pd.DataFrame([processed])[expected_features[target_variable]]
    
    return processed_df.to_numpy().reshape(1, -1)

def predict_score(input_data, target_variable):
    """Use the trained model to make predictions"""
    processed_data = preprocess_input(input_data, target_variable)
    prediction = model.predict(processed_data)[0]
    
    # Ensure the score is within 0-100 range
    if prediction < 0:
        return 0.0
    elif prediction > 100:
        return 100.0
    else:
        return round(prediction, 2)

# Create form for user input
with st.form("student_info_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Student Demographics")
        gender = st.radio("Gender", ["male", "female"])
        ethnic_group = st.selectbox("Ethnic Group", ["group A", "group B", "group C", "group D", "group E"])
        parent_educ = st.selectbox("Parent's Education", 
                                  ["master's degree", "bachelor's degree", "associate's degree", 
                                   "some college", "high school", "some high school"])
        lunch_type = st.radio("Lunch Type", ["standard", "free/reduced"])
        test_prep = st.radio("Test Preparation", ["completed", "none"])
        
    with col2:
        st.subheader("Family & Study Information")
        parent_marital_status = st.selectbox("Parent Marital Status", ["married", "single", "divorced", "widowed"])
        practice_sport = st.selectbox("Practice Sport", ["never", "sometimes", "regularly"])
        is_first_child = st.radio("Is First Child", ["yes", "no"])
        transport_means = st.radio("Transportation Means", ["private", "school_bus", "public"])
        wkly_study_hours = st.selectbox("Weekly Study Hours", ["> 10", "5 - 10", "< 5"])
        
        st.subheader("Existing Scores")
        # Show different score inputs based on selected model
        if model_option == "Writing Score Prediction":
            math_score = st.slider("Math Score", 0, 100, 70)
            reading_score = st.slider("Reading Score", 0, 100, 75)
        elif model_option == "Math Score Prediction":
            reading_score = st.slider("Reading Score", 0, 100, 75)
            writing_score = st.slider("Writing Score", 0, 100, 75)
        else:  # Reading Score Prediction
            math_score = st.slider("Math Score", 0, 100, 70)
            writing_score = st.slider("Writing Score", 0, 100, 75)
    
    # Submit button
    submitted = st.form_submit_button(f"Predict {target_variable}")

# When form is submitted
if submitted:
    # Create input dictionary based on selected model
    input_dict = {
        'Gender': gender,
        'EthnicGroup': ethnic_group,
        'ParentEduc': parent_educ,
        'LunchType': lunch_type,
        'TestPrep': test_prep,
        'ParentMaritalStatus': parent_marital_status,
        'PracticeSport': practice_sport,
        'IsFirstChild': is_first_child,
        'TransportMeans': transport_means,
        'WklyStudyHours': wkly_study_hours
    }
    
    # Add appropriate scores based on model
    if model_option == "Writing Score Prediction":
        input_dict['MathScore'] = math_score
        input_dict['ReadingScore'] = reading_score
    elif model_option == "Math Score Prediction":
        input_dict['ReadingScore'] = reading_score
        input_dict['WritingScore'] = writing_score
    else:  # Reading Score Prediction
        input_dict['MathScore'] = math_score
        input_dict['WritingScore'] = writing_score
    
    input_df = pd.DataFrame([input_dict])
    
    # Display the user input
    st.subheader("Student Information Summary")
    st.dataframe(input_df)
    
    # Show the encoded values (for debugging)
    with st.expander("View Encoded Values (For Debugging)"):
        encoded_data = preprocess_input(input_dict, target_variable)
        st.write("Encoded features:", expected_features[target_variable])
        st.write("Encoded values:", encoded_data[0])
    
    # Make prediction
    try:
        predicted_score = predict_score(input_dict, target_variable)
        
        # Display prediction
        st.subheader("Prediction Result")
        st.metric(label=f"Predicted {target_variable}", value=f"{predicted_score:.2f}")
        
        # Additional visualization
        st.subheader("Score Comparison")
        
        # Create comparison data based on model type
        if model_option == "Writing Score Prediction":
            scores_data = pd.DataFrame({
                'Subject': ['Math', 'Reading', 'Writing'],
                'Score': [math_score, reading_score, predicted_score]
            })
        elif model_option == "Math Score Prediction":
            scores_data = pd.DataFrame({
                'Subject': ['Reading', 'Writing', 'Math'],
                'Score': [reading_score, writing_score, predicted_score]
            })
        else:  # Reading Score Prediction
            scores_data = pd.DataFrame({
                'Subject': ['Math', 'Writing', 'Reading'],
                'Score': [math_score, writing_score, predicted_score]
            })
        
        st.bar_chart(scores_data.set_index('Subject'))
        
        # Add some interpretation
        if predicted_score >= 80:
            st.info("🎉 Excellent performance predicted!")
        elif predicted_score >= 60:
            st.info("👍 Good performance predicted.")
        else:
            st.warning("💡 This area may need improvement. Consider additional study support.")
        
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

# Add some information about the model
with st.expander("About This Model"):
    st.markdown(f"""
    **{model_option}**
    
    This prediction model was trained on student performance data with the following features:
    - **Demographics**: Gender, Ethnic Group
    - **Family Background**: Parent's Education, Parent Marital Status, Is First Child
    - **School Factors**: Lunch Type, Test Preparation, Transportation Means
    - **Study Habits**: Weekly Study Hours, Practice Sport
    - **Existing Scores**: {', '.join([f for f in expected_features[target_variable] if 'Score' in f and f != target_variable])}
    
    The model predicts the {target_variable.lower()} based on these input features.
    
    **Model file location**: {MODEL_PATHS[target_variable]}
    """)

# Footer
st.markdown("---")
st.markdown("© 2023 Student Performance Prediction Dashboard")