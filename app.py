
import streamlit as st
import joblib
import pandas as pd

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

with st.sidebar:
    st.title("Menu")
    st.caption("⚠️ این ابزار جایگزین تشخیص و مشاوره‌ی پزشکی نیست.")

st.title("heart_disase")

with st.form("prediction_form", border=True):

    Age = st.number_input("Enter your age :", min_value=1, value=50)

    sex = st.radio("Select Gender (man = 1 , female = 0) : ", [1, 0])

    # -----------------------------------------chest pain type-------------------------------------------
    cp = st.slider(
        "Chest pain type:",
        min_value=1,
        max_value=4
    )

    st.info(
        """
    1 = Typical Angina (آنژین معمولی)

    2 = Atypical Angina (آنژین غیرمعمولی)

    3 = Non-anginal Chest Pain (درد غیرآنژینی)

    4 = Asymptomatic (بدون علامت)"""
    )
    # -------------------------------------------Resting blood pressure----------------------------------------------

    trestbps = st.number_input("Enter your Resting blood pressure :",
        min_value=0,
        value=110,
        step=10
        )

    # ----------------------------------------------------Serum cholesterol------------------------------------------------------------------------

    chol = st.number_input("what is your Serum cholesterol number ?",
        min_value=0,
        value=200
        )

    # ------------------------------------Fasting blood sugar > 120 mg/dl------------------------------------------------------
    fbs = st.radio("do you have Fasting blood sugar > 120 mg/dl ? (yes=1 , no=0)", [0, 1])
    # --------------------------------------------------Resting electrocardiographic result---------------------------------------------------------------

    restecg = st.radio("Resting electrocardiographic result :", [0, 1, 2])
    st.info(
        """
    0 = Normal

    1 = ST-T wave abnormality

    2 = Probable left ventricular hypertrophy"""
    )
    # ---------------------------------------------Maximum heart rate achieved during exercise-----------------------------------------------
    thalach = st.number_input("Maximum heart rate achieved during exercise :",
        min_value=0,
        value=90
        )

    # ----------------------------------Exercise-induced angina----------------------------------------------------------
    exang = st.radio("do you have  exang ? (yes = 1 , no = 0)", [0, 1])
    # ----------------------------------------ST depression induced by exercise relative to rest-----------------------------------------
    oldpeak = st.number_input("ST depression induced by exercise relative to rest:",
        min_value=0.0,
        step=0.1
        )
    # -----------------------------------------------Slope of the ST segment during peak exercise------------------------------------------------------

    slope = st.selectbox("Select a Slope of the ST segment during peak exercise:", [1, 2, 3])

    st.info(
        """
    1 =Upsloping

    2= Flat

    3 =Downsloping"""
    )
    # -------------------------------------Number of major vessels colored by fluoroscopy---------------------------------------------------------------
    Ca = st.number_input("Number of major vessels colored by fluoroscopy:",
        min_value=0,
        max_value=3,
        value=0
        )

    # -------------------------------------------Thallium stress test result-----------------------------------------------------
    thal = st.selectbox("Select a Thallium stress test result : ", [3, 6, 7])
    st.info(
        """
        3 = Normal

        6 = Fixed Defect

        7 = Reversible Defect  """
    )

    submitted = st.form_submit_button("پیش‌بینی ریسک")


if submitted:
    # ---------- لود مدل، اسکیلر، و اسم ستون‌ها ----------
    model = joblib.load('heart_disease_model.pkl')
    scaler = joblib.load('scaler.pkl')
    feature_columns = joblib.load('feature_columns.pkl')

    # ---------- ساخت دیتافریم ورودی از داده‌های خام کاربر ----------
    input_dict = {
        'age': Age,
        'sex': sex,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': bool(fbs),
        'thalach': thalach,
        'exang': bool(exang),
        'oldpeak': oldpeak
    }
    input_df = pd.DataFrame([input_dict])

    # ---------- اضافه کردن ستون‌های One-Hot Encoding مطابق مدل ----------
    for col in feature_columns:
        if col.startswith('cp_') and col == f'cp_{cp}':
            input_df[col] = True
        elif col.startswith('thal_') and col == f'thal_{float(thal)}':
            input_df[col] = True
        elif col.startswith('ca_') and col == f'ca_{float(Ca)}':
            input_df[col] = True
        elif col.startswith('slope_') and col == f'slope_{float(slope)}':
            input_df[col] = True
        elif col.startswith('restecg_') and col == f'restecg_{restecg}':
            input_df[col] = True

    # ---------- پر کردن بقیه‌ی ستون‌های Encoding با False ----------
    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = False

    # ---------- مرتب‌سازی دقیق مطابق ترتیب train ----------
    input_df = input_df[feature_columns]

    # ---------- Scale کردن فیچرهای عددی ----------
    numeric_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    input_df[numeric_features] = scaler.transform(input_df[numeric_features])

    # ---------- پیش‌بینی ----------
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.markdown("---")
    st.subheader("نتیجه‌ی پیش‌بینی")

    if probability < 0.3:
        st.success(f"✅ ریسک پایین بیماری قلبی: {probability*100:.1f}%")
    elif probability < 0.7:
        st.warning(f"⚠️ ریسک متوسط بیماری قلبی: {probability*100:.1f}% — مشورت با پزشک توصیه می‌شود")
    else:
        st.error(f"🔴 ریسک بالای بیماری قلبی: {probability*100:.1f}% — مراجعه به پزشک توصیه می‌شود")