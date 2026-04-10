import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="wide"
)

# ---------------------------
# CUSTOM CSS (PREMIUM UI)
# ---------------------------
st.markdown("""
<style>

/* SAFE BACKGROUND */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(to right, #1e3c72, #2a5298);
}

/* FIX LAYOUT */
section.main > div {
    padding-top: 2rem;
}

/* Button */
.stButton>button {
    background: linear-gradient(to right, #00c6ff, #0072ff);
    color: white;
    font-size: 18px;
    border-radius: 12px;
    height: 50px;
    width: 100%;
}

/* Success box */
.success-box {
    background: #d4edda;
    padding: 20px;
    border-radius: 10px;
    color: #155724;
}

/* Error box */
.error-box {
    background: #f8d7da;
    padding: 20px;
    border-radius: 10px;
    color: #721c24;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------
# TITLE
# ---------------------------
st.markdown("<h1 style='text-align:center;'>🩺 Diabetes Prediction Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>AI-powered health risk analysis</p>", unsafe_allow_html=True)

# ---------------------------
# LOAD DATA
# ---------------------------
@st.cache_data
def load_data():
    return pd.read_csv("diabetes_prediction_dataset.csv")

df = load_data()

# ---------------------------
# PREPROCESSING
# ---------------------------
le_gender = LabelEncoder()
le_smoking = LabelEncoder()

df['gender'] = le_gender.fit_transform(df['gender'].astype(str))
df['smoking_history'] = le_smoking.fit_transform(df['smoking_history'].astype(str))

X = df.drop("diabetes", axis=1)
y = df["diabetes"]

# ---------------------------
# MODEL
# ---------------------------
@st.cache_resource
def train_model():
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)
    return model

model = train_model()

# ---------------------------
# LAYOUT
# ---------------------------
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 👤 Patient Info")
    gender = st.selectbox("Gender", le_gender.classes_)
    age = st.slider("Age", 1, 100, 30)
    hypertension = st.selectbox("Hypertension", [0, 1])
    heart_disease = st.selectbox("Heart Disease", [0, 1])

with col2:
    st.markdown("### 🧪 Medical Data")
    smoking = st.selectbox("Smoking History", le_smoking.classes_)
    bmi = st.number_input("BMI", 10.0, 60.0, 25.0)
    hba1c = st.number_input("HbA1c Level", 3.0, 15.0, 5.5)
    glucose = st.number_input("Blood Glucose Level", 50, 300, 120)

# ---------------------------
# PREPARE INPUT
# ---------------------------
gender_enc = le_gender.transform([gender])[0]
smoking_enc = le_smoking.transform([smoking])[0]

input_data = np.array([[gender_enc, age, hypertension, heart_disease,
                        smoking_enc, bmi, hba1c, glucose]])

# ---------------------------
# PREDICTION
# ---------------------------

st.markdown("---")

if st.button("🚀 Predict Now"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.markdown("## 📊 Result")

    # Progress bar
    st.progress(int(probability * 100))

    if prediction == 1:
        st.markdown(f"""
        <div class="error-box">
        <h3>⚠️ High Risk of Diabetes</h3>
        <p>Probability: {probability*100:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("💡Recommendations")
        st.write("""
- 🥗 Healthy diet  
- 🏃 Regular exercise  
- 📉 Reduce sugar  
- 🩺 Consult doctor  
""")

        st.markdown("🥗 Personalized Diet Plan (High Risk)")
        st.markdown("""
🌅 Morning (Empty Stomach)
- Warm water with lemon 🍋
- 5 soaked almonds 🌰

🍳 Breakfast
- Oats / Multigrain toast
- Boiled eggs or sprouts
- Green tea 🍵

🍎 Mid-Morning Snack
- Apple / Guava
- Coconut water 🥥

 🍛 Lunch
- 2 chapati (whole wheat)
- Brown rice (small portion)
- Dal + green vegetables 🥦
- Salad 🥗

☕ Evening Snack
- Roasted chana / nuts
- Herbal tea

#### 🍽 Dinner
- Soup + vegetables
- Grilled paneer / chicken

🚫 Avoid
- Sugar, sweets 🍬
- Fried food 🍟
- Soft drinks 🥤
""")

    else:
        st.markdown(f"""
        <div class="success-box">
        <h3>✅ Low Risk of Diabetes</h3>
        <p>Probability: {probability*100:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("👍 Keep it up!")
        st.write("""
- 💪 Stay active  
- 🍎 Balanced diet  
- ⚖️ Maintain weight  
""")

        st.markdown("🥗 Recommended Diet Plan (Maintain Health)")
        st.markdown("""
🌅 Morning
- Warm water + honey 🍯
- Fresh fruits 🍊

🍳 Breakfast
- Eggs / Poha / Upma
- Milk or smoothie 🥛

🍎 Mid Snack
- Fruits or dry fruits

🍛 Lunch
- Rice + chapati
- Dal + vegetables 🥬
- Curd

☕ Evening
- Tea + light snacks

 🍽 Dinner
- Light meal (chapati + sabzi)

💡 Tips
- Stay active 🏃
- Drink water 💧
- Avoid junk food 🍔
""")

# ---------------------------
# FOOTER
# ---------------------------
st.markdown("---")