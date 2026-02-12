import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load data with caching
@st.cache_data
def load_data():
    df = pd.read_csv('cleaned_ayurdata.csv')
    try:
        symptoms_list = pd.read_csv('symptom_list.csv')
    except:
        # Fallback symptom list if CSV missing
        symptoms_list = pd.DataFrame({
            'symptom': ['Cough', 'Sore throat', 'Fever', 'Fatigue', 'Joint pain', 
                       'Headache', 'Nausea', 'Chest congestion', 'Breathing difficulty',
                       'Swelling', 'Stress', 'High blood pressure', 'Frequent urination']
        })
    return df, symptoms_list

df, symptoms_list = load_data()

# Page config
st.set_page_config(page_title="AyurVaidya Assist", layout="wide", initial_sidebar_state="expanded")

# Header
st.title("🪔 AyurVaidya Assist")
st.markdown("**AI-Powered Ayurvedic Symptom Analysis & Remedy Guide**")
st.markdown("---")

# Sidebar disclaimer
with st.sidebar:
    st.error("❗ **NOT medical advice** - Always consult a doctor")
    st.markdown("**How it works:** Select symptoms → AI finds best matches → Ayurvedic remedies")

# Safety checks (unique keys)
seen_doctor = st.sidebar.checkbox("✅ Already seeing doctor?", key="doctor_status")
emergency = st.sidebar.checkbox("🚨 Emergency symptoms?", key="emergency_status")

if seen_doctor:
    st.sidebar.error("👨‍⚕️ Follow your doctor's advice first")
    st.stop()
if emergency:
    st.error("🚨 **EMERGENCY** - Seek immediate medical help!")
    st.stop()

# Main symptom selector - FIXED UNIQUE KEYS
st.header("📋 Step 1: Select Your Symptoms")
st.info("👆 Choose 2-3 symptoms you are experiencing")

selected_symptoms = []
for idx, row in symptoms_list.iterrows():
    symptom = row['symptom'].strip()
    if st.checkbox(symptom, key=f"chk_{idx}_{hash(symptom)}"):
        selected_symptoms.append(symptom)

# Validate input
if len(selected_symptoms) < 1:
    st.warning("⚠️ Please select at least **2 symptoms** for accurate analysis")
    st.stop()

st.success(f"✅ Analyzing **{len(selected_symptoms)}** symptoms: **{', '.join(selected_symptoms[:3])}{'...' if len(selected_symptoms)>3 else ''}**")

# AI Analysis Section
st.header("🔬 Step 2: AI Analysis")
progress_bar = st.progress(0)
status_text = st.empty()

# Prepare text for matching
df['match_text'] = (
    df['symptoms'].fillna('').astype(str) + ' ' +
    df['risk_factors'].fillna('').astype(str) + ' ' +
    df['environmental_factors'].fillna('').astype(str)
)

# TF-IDF similarity matching
with st.spinner("Computing Ayurvedic matches..."):
    tfidf = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1,2))
    disease_vectors = tfidf.fit_transform(df['match_text'])
    user_query = ' '.join(selected_symptoms)
    user_vector = tfidf.transform([user_query])
    similarities = cosine_similarity(user_vector, disease_vectors)[0]
    
    # Top 3 matches
    top_indices = np.argsort(similarities)[-3:][::-1]
    progress_bar.progress(100)

st.success("✅ **Analysis Complete!** Here are your top matches:")

# Results Display
for i, idx in enumerate(top_indices):
    similarity_score = similarities[idx]
    if similarity_score > 0.05:  # Show matches above 5%
        disease_row = df.iloc[idx]
        
        # Main result row
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.markdown(f"**{i+1}. {disease_row['disease']}**")
            st.caption(f"💡 Matching symptoms: {disease_row['symptoms'][:100]}...")
        with col2:
            st.metric("Match", f"{similarity_score:.0%}")
        with col3:
            st.caption(f"Severity: {disease_row['symptom_severity']}")
        
        # Detailed treatment plan
        with st.expander(f"🌿 View Complete Ayurvedic Treatment Plan", expanded=(i==0)):
            
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("💊 Ayurvedic Remedy")
                if pd.notna(disease_row['ayurvedic_herbs']):
                    st.info(f"**🌱 Herbs**: {disease_row['ayurvedic_herbs']}")
                if pd.notna(disease_row['formulation']):
                    st.success(f"**🥄 Recipe**: {disease_row['formulation']}")
                if pd.notna(disease_row['duration_of_treatment']):
                    st.info(f"**⏱️ Duration**: {disease_row['duration_of_treatment']}")
            
            with c2:
                st.subheader("🏃‍♂️ Lifestyle Changes")
                if pd.notna(disease_row['yoga__physical_therapy']):
                    st.success(f"**🧘 Yoga**: {disease_row['yoga__physical_therapy']}")
                if pd.notna(disease_row['diet_and_lifestyle_recommendations']):
                    st.info(f"**🍎 Diet**: {disease_row['diet_and_lifestyle_recommendations'][:200]}...")
            
            # Additional info
            if pd.notna(disease_row['environmental_factors']):
                st.warning(f"**🌍 Environment**: {disease_row['environmental_factors']}")
            
            if pd.notna(disease_row['prevention']):
                st.info(f"**🛡️ Prevention**: {disease_row['prevention']}")

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**✅ Why Ayurveda?**")
    st.caption("- Natural ingredients\n- Targets root cause\n- Minimal side effects")
with col2:
    st.markdown("**📊 Data**")
    st.caption(f"446 diseases analyzed")
with col3:
    st.markdown("**🔬 Powered by**")
    st.caption("AyurGenixAI Dataset")
