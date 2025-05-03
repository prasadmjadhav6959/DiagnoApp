# diagno_app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import lzma

# --- Streamlit Config ---
st.set_page_config(page_title="Diagno Medical Healthcare", layout="centered")

# --- Sidebar Navigation ---
st.sidebar.title("Diagno Services")
section = st.sidebar.radio("Go to", [
    "Test Result Time Estimator",
    "Treatment Cost Predictor",
    "Treatment Result Predictor",
    "Test Duration Estimator"
])

# --- Common Header ---
st.markdown("<h1 style='color:#77D84C;'>Diagno Medical Healthcare Solution Services Provider</h1>", unsafe_allow_html=True)
st.image("diagno.png", use_column_width=True)

# --- Load Model Helper ---
@st.cache_resource
def load_model(filename):
    try:
        with lzma.open(filename, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# --- 1. Test Result Time Estimator ---
if section == "Test Result Time Estimator":
    st.markdown("<h3 style='color:#77D84C;'>Test Result Time Estimator</h3>", unsafe_allow_html=True)

    tp_model = load_model('test_time_predictor.pkl.xz')
    t_c_dt = pd.read_csv('t_c_dt.csv')
    t_n_dt = pd.read_csv('t_n_dt.csv')
    cat_cols = ['gender', 'test_type', 'follow_up_required', 'symptoms', 'previous_conditions']
    encoded_dict = {col: dict(zip(t_c_dt[col], t_n_dt[col])) for col in cat_cols}

    hospital_id = st.number_input("Enter Hospital ID", min_value=0, step=1)
    selected_vals = {}
    for col in cat_cols:
        val = st.selectbox(f'Select {col.replace("_", " ").capitalize()}', t_c_dt[col].unique(), key=col)
        selected_vals[col] = encoded_dict[col][val]

    if st.button("Estimate Test Result Time"):
        try:
            input_point = np.array([[selected_vals['gender'], selected_vals['test_type'],
                                     hospital_id, selected_vals['follow_up_required'],
                                     selected_vals['symptoms'], selected_vals['previous_conditions']]])
            prediction = tp_model.predict(input_point)
            st.success(f"🕒 Estimated Test Result Time: {prediction[0]:.2f} units")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# --- 2. Treatment Cost Predictor ---
elif section == "Treatment Cost Predictor":
    st.markdown("<h3 style='color:#77D84C;'>Clinical Cost Predictor</h3>", unsafe_allow_html=True)

    reg_lr_model = load_model('treatment_cost_predictor.pkl.xz')
    tc_c_df = pd.read_csv('tc_c_df.csv')
    tc_n_df = pd.read_csv('tc_n_df.csv')
    cat_cols = ['gender', 'test_type', 'treatment_plan', 'symptoms', 'previous_conditions']
    encoded_dict = {col: dict(zip(tc_c_df[col], tc_n_df[col])) for col in cat_cols}

    st.number_input("Enter Patient Age", min_value=0, max_value=120, step=1, key="age")
    selected_vals = {}
    for col in cat_cols:
        val = st.selectbox(f'Select {col}', tc_c_df[col].unique(), key=col)
        selected_vals[col] = encoded_dict[col][val]

    insurance = st.selectbox("Do you have Insurance Coverage?", ["No", "Yes"])
    insurance_encoded = 0 if insurance == "No" else 1

    input_point = np.array([[selected_vals['gender'], selected_vals['test_type'], selected_vals['treatment_plan'],
                             selected_vals['symptoms'], selected_vals['previous_conditions'], insurance_encoded]])

    if st.button("Predict Treatment Cost"):
        prediction = reg_lr_model.predict(input_point)
        st.success(f"🩺 Estimated Treatment Cost: ₹ {prediction[0]:,.2f}")

# --- 3. Treatment Result Predictor ---
elif section == "Treatment Result Predictor":
    st.markdown("<h3 style='color:#77D84C;'>Patients Treatment Predictor (Success/Pending)</h3>", unsafe_allow_html=True)

    clf_lr_model = load_model('treatment_result_predictor.pkl.xz')
    pt_c_df = pd.read_csv('pt_c_df.csv')
    pt_n_df = pd.read_csv('pt_n_df.csv')
    cat_cols = ['gender', 'diagnosis', 'treatment_plan', 'symptoms', 'previous_conditions', 'smoking_status', 'physical_activity_level']
    encoded_dict = {col: dict(zip(pt_c_df[col], pt_n_df[col])) for col in cat_cols}

    age = st.number_input("Enter Patient Age", min_value=0, max_value=120, step=1)
    test_result = st.number_input("Enter Test Result (numeric)", min_value=0.0, step=0.1)

    selected_vals = {}
    for col in cat_cols:
        val = st.selectbox(f'Select {col}', pt_c_df[col].unique(), key=col)
        selected_vals[col] = encoded_dict[col][val]

    input_data = np.array([[age, test_result] + list(selected_vals.values())])

    if st.button("Predict Treatment Success", key="predict_success"):
        prediction = clf_lr_model.predict(input_data)
        if prediction[0] == 1:
            st.success("✅ Treatment is Predicted to be Successful")
        else:
            st.error("❌ Treatment is Predicted to be Unsuccessful")

# --- 4. Test Duration Estimator ---
elif section == "Test Duration Estimator":
    st.markdown("<h3 style='color:#77D84C;'>Test Duration Estimator</h3>", unsafe_allow_html=True)

    td_model = load_model('test_duration_predictor.pkl.xz')
    td_c = pd.read_csv('td_c.csv')
    td_n = pd.read_csv('td_n.csv')
    cat_cols = ['hospital_id', 'doctor_id', 'test_type']
    encoded_dict = {col: dict(zip(td_c[col], td_n[col])) for col in cat_cols}

    selected_vals = {}
    for col in cat_cols:
        val = st.selectbox(f"Select {col.replace('_', ' ').capitalize()}", td_c[col].unique(), key=col)
        selected_vals[col] = encoded_dict[col][val]

    if st.button("Estimate Test Duration"):
        try:
            input_array = np.array([[selected_vals['hospital_id'], selected_vals['doctor_id'], selected_vals['test_type']]])
            prediction = td_model.predict(input_array)
            st.success(f"⌛ Estimated Test Duration: {prediction[0]:.2f} hours")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            
# ---------------------------------------------
# Description Section
# ---------------------------------------------
st.write("---")
st.header("📄 Description")
st.markdown("""
This Diagno Medical Healthcare platform provides predictive insights for:

- 🕒 **Test Result Time Estimation**
- ⌛ **Test Duration Estimation**
- 💰 **Treatment Cost Estimation**
- ✅ **Treatment Success Prediction**

Users can enter medical test and patient-related details to receive predictions powered by trained machine learning models. This tool aims to support healthcare professionals and patients by enhancing decision-making accuracy and efficiency.
""")

# ---------------------------------------------
# Selected Values Overview
# Selected Input Summary (JSON Display)
# ---------------------------------------------
st.write("---")
st.subheader("🔍 Selected Input Summary")

selected_encoded_values = [
    {
        "patient_id": "0100072",
        "age": 40,
        "gender": "Female",
        "test_type": "X-Ray",
        "test_result": "Positive", 
        "diagnosis": "Negative",
        "treatment_plan": "Medication",
        "hospital_id": "20002",
        "test_date": "2023-04-04",
        "result_date": "2023-04-08",
        "doctor_id": "50001",
        "test_duration": 171,
        "treatment_cost": 3289,
        "follow_up_required": "No",
        "symptoms": "Headache",
        "previous_conditions": "Hypertension",
        "insurance_coverage": 35,
        "family_history": "Yes",
        "body_mass_index": 28.22,
        "smoking_status": "Smoker",
        "alcohol_consumption": "Yes",
        "physical_activity_level": "High",
        "test_result_normalized": 0.427478,
        "treatment_success": "Unsuccessful"
    }
]

st.json(selected_encoded_values)

# ---------------------------------------------
# Feedback Form
# ---------------------------------------------
st.write("---")
st.header("📝 We Value Your Feedback")
with st.form("feedback_form"):
    name = st.text_input("Your Name")
    email = st.text_input("Your Email")
    feedback = st.text_area("Your Feedback or Suggestions")
    submitted = st.form_submit_button("Submit")

    if submitted:
        st.success("🎉 Thank you for your feedback! We’ll use it to improve our healthcare services.")
        # Optional: Save to database, log, or email logic goes here
        
# LinkedIn: linkedin.com/in/prasadmjadhav2 | Github: github.com/prasadmjadhav2 | Mail: prasadmjadhav6161@gmail.com