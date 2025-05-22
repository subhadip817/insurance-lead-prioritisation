import streamlit as st
import pandas as pd
import joblib

# Title
st.title("🔮 Smart Lead Prioritization – Streamlit App")

# Load model and encoders
@st.cache_resource
def load_model():
    model = joblib.load("lead_scoring_model.pkl")
    encoders = joblib.load("label_encoders.pkl")
    return model, encoders

model, le_dict = load_model()

# File uploader
uploaded_file = st.file_uploader("Upload a CSV with leads for scoring", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Uploaded Data Preview")
    st.write(df.head())

    df = df.dropna()

    # Encode categorical columns
    for col in ['Occupation', 'Source']:
        if col in df.columns:
            df[col] = le_dict[col].transform(df[col])

    # Predict probabilities
    probabilities = model.predict_proba(df)[:, 1]

    def assign_priority(p):
        if p > 0.75:
            return 'Hot'
        elif p > 0.4:
            return 'Warm'
        else:
            return 'Cold'

    df['Predicted_Probability'] = probabilities
    df['Priority_Bucket'] = df['Predicted_Probability'].apply(assign_priority)

    st.subheader("🔥 Scored Leads")
    st.dataframe(df[['Predicted_Probability', 'Priority_Bucket']].head(10))

    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Scored Leads CSV", csv, "scored_leads.csv", "text/csv")
