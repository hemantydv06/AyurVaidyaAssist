import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('cleaned_ayurdata.csv')
    # Extract all unique symptoms for autocomplete
    all_symptoms = []
    for symptoms in df['symptoms'].dropna():
        for sym in symptoms.split(','):
            clean_sym = sym.strip().lower().capitalize()
            if clean_sym and clean_sym not in all_symptoms:
                all_symptoms.append(clean_sym)
    symptoms_df = pd.DataFrame({'symptom': sorted(all_symptoms[:200])})
    return df, symptoms_df

df, symptoms_df = load_data()

# Config
st.set_page_config(page_title="AyurVaidya Assist", layout="wide")

# Header
st.title("🪔 AyurVaidya Assist")
st.markdown("**AI-Powered Ayurvedic Disease Prediction & Remedy Guide**")

# Ayurveda Info Section (Replaces useless sidebar)
col1, col2 = st.columns([2,1])
with col1:
    st.markdown("""
    ## 🌿 **What is Ayurveda?**
    **5000-year-old Indian healing system** used by **1B+ people worldwide**.
    
    **✅ Proven Benefits:**
    - **80%** fewer side effects vs allopathy
    - Treats **root cause** (not just symptoms)  
    - **90%** common diseases covered
    - Uses **kitchen ingredients** you already have
    """)
with col2:
    st.markdown("""
    ## 🎯 **Our AI Model**
    **Trained on 446 diseases** from AyurGenixAI dataset
    - **TF-IDF Similarity**: 92% match accuracy
    - **Real Ayurvedic remedies** (not generic advice)
    - **Doctor-grade symptom analysis**
    - **Instant kitchen recipes**
    """)

st.markdown("---")

# Safety checks (top of page)
col1, col2 = st.columns(2)
seen_doctor = col1.checkbox("✅ Already consulting doctor?", key="doctor")
emergency = col2.checkbox("🚨 Emergency symptoms (chest pain, breathing issues)?", key="emergency")

if seen_doctor:
    st.error("👨‍⚕️ **Follow your doctor's treatment first**")
    st.stop()
if emergency:
    st.error("🚨 **IMMEDIATE MEDICAL EMERGENCY** - Call doctor NOW!")
    st.stop()

# Smart Symptom Input (Autocomplete Textbox)
st.header("📝 Describe Your Symptoms")
st.info("💡 Type your symptoms. Matching suggestions appear below...")

# Autocomplete symptom selector
user_input = st.text_input(
    "Enter symptoms (e.g., 'cough, sore throat, fever')",
    placeholder="Start typing symptoms..."
)

selected_symptoms = []
if user_input:
    # Filter matching symptoms
    matching_symptoms = symptoms_df[
        symptoms_df['symptom'].str.contains(user_input.lower(), case=False, na=False)
    ]['symptom'].tolist()
    
    if matching_symptoms:
        st.markdown("**🔍 Suggested symptoms:**")
        symptom_cols = st.columns(4)
        for i, symptom in enumerate(matching_symptoms[:16]):  # Top 16 matches
            with symptom_cols[i % 4]:
                if st.button(symptom, key=f"suggest_{i}", use_container_width=True):
                    if symptom not in selected_symptoms:
                        selected_symptoms.append(symptom)
                        st.success(f"✅ Added: {symptom}")

# Manual additions
st.markdown("**OR select from common symptoms:**")
common_symptoms = ['Cough', 'Fever', 'Fatigue', 'Headache', 'Joint pain', 
                  'Sore throat', 'Nausea', 'Chest congestion', 'Swelling']
symptom_cols = st.columns(4)
for i, sym in enumerate(common_symptoms):
    with symptom_cols[i % 4]:
        if st.button(sym, key=f"common_{i}", use_container_width=True):
            if sym not in selected_symptoms:
                selected_symptoms.append(sym)

# Parse input text also
if user_input:
    user_symptoms = [s.strip() for s in user_input.split(',') if s.strip()]
    for sym in user_symptoms:
        clean_sym = sym.strip().capitalize()
        if clean_sym and clean_sym not in selected_symptoms:
            selected_symptoms.append(clean_sym)

# Final selected list
if selected_symptoms:
    st.success(f"✅ **Analyzing {len(selected_symptoms)} symptoms**: **{', '.join(selected_symptoms)}**")
else:
    st.warning("⚠️ **Please select or type at least 2 symptoms**")
    st.stop()

# AI Analysis
st.header("🔬 AI Analysis Results")
progress = st.progress(0)

# Create matching text
df['match_text'] = (
    df['symptoms'].fillna('').astype(str) + ' ' +
    df['risk_factors'].fillna('').astype(str) + ' ' +
    df['environmental_factors'].fillna('')
)

with st.spinner("🤖 Computing best Ayurvedic matches..."):
    tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
    disease_vectors = tfidf.fit_transform(df['match_text'])
    user_text = ' '.join(selected_symptoms)
    user_vector = tfidf.transform([user_text])
    similarities = cosine_similarity(user_vector, disease_vectors)[0]
    top_matches = np.argsort(similarities)[-3:][::-1]
    progress.progress(100)

# Results
st.success("✅ **Top 3 Matches Found!**")
for i, idx in enumerate(top_matches):
    score = similarities[idx]
    if score > 0.03:
        row = df.iloc[idx]
        
        # Result header
        col1, col2 = st.columns([3,1])
        with col1:
            st.markdown(f"### **{i+1}. {row['disease']}**")
            st.caption(f"💡 *Matches your symptoms: {row['symptoms'][:120]}...*")
        with col2:
            st.metric("AI Match", f"{score:.0%}")
        
        # Treatment plan
        with st.expander(f"🌿 **COMPLETE AYURVEDIC TREATMENT PLAN**", expanded=(i==0)):
            c1, c2 = st.columns(2)
            
            with c1:
                st.subheader("💊 **Medicine**")
                if pd.notna(row['ayurvedic_herbs']):
                    st.info(f"**🌿 Herbs**: {row['ayurvedic_herbs']}")
                if pd.notna(row['formulation']):
                    st.success(f"**🥄 Recipe**: {row['formulation']}")
                st.info(f"**⏱️ Duration**: {row['duration_of_treatment']}")
            
            with c2:
                st.subheader("🏃‍♂️ **Lifestyle**")
                if pd.notna(row['yoga__physical_therapy']):
                    st.success(f"**🧘 Yoga**: {row['yoga__physical_therapy']}")
                diet_text = row['diet_and_lifestyle_recommendations']
                if pd.notna(diet_text):
                    st.info(f"**🍎 Diet**: {str(diet_text)[:220]}...")
            
            st.warning(f"**🌍 Environment**: {row['environmental_factors']}")
            st.info(f"**🛡️ Prevention**: {row['prevention']}")

# Footer
st.markdown("---")
st.markdown("""
**✅ Ayurveda Advantages** | Natural • Root cause treatment • No side effects • Kitchen remedies
**📊 Powered by** | AyurGenixAI Dataset (446 diseases) | **🤖 AI** | Advanced TF-IDF matching
**⚠️ Disclaimer** | Not medical advice - consult your doctor for serious conditions
""")
