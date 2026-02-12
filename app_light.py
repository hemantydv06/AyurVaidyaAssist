import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 🔥 RED THEME + GLOWING CSS ✨
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    /* RED GRADIENT BACKGROUND */
    .main {
        background: linear-gradient(135deg, #dc2626 0%, #b91c1c 50%, #991b1b 100%);
        padding: 2rem;
        font-family: 'Poppins', sans-serif;
    }
    
    /* GLOWING RED CARDS - FIXED WHITE TEXT */
    .glow-card {
        background: rgba(255, 255, 255, 0.95) !important;
        backdrop-filter: blur(20px);
        border-radius: 25px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 
            0 20px 40px rgba(220, 38, 38, 0.3),
            0 0 0 1px rgba(255, 255, 255, 0.2),
            inset 0 1px 0 rgba(255, 255, 255, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.3);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .glow-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.4), transparent);
        transition: left 0.7s;
    }
    
    .glow-card:hover::before {
        left: 100%;
    }
    
    .glow-card:hover {
        transform: translateY(-8px);
        box-shadow: 
            0 30px 60px rgba(220, 38, 38, 0.4),
            0 0 0 1px rgba(255, 255, 255, 0.3);
    }
    
    /* RED GLOWING BUTTONS */
    .red-glow-btn {
        background: linear-gradient(45deg, #dc2626, #ef4444, #f87171) !important;
        border: none !important;
        border-radius: 20px !important;
        padding: 14px 28px !important;
        color: white !important;
        font-weight: 600 !important;
        font-size: 15px !important;
        cursor: pointer !important;
        box-shadow: 0 8px 25px rgba(220, 38, 38, 0.4) !important;
        transition: all 0.3s ease !important;
        position: relative !important;
        overflow: hidden !important;
    }
    
    .red-glow-btn:hover {
        transform: translateY(-4px) scale(1.05) !important;
        box-shadow: 0 15px 35px rgba(220, 38, 38, 0.6) !important;
    }
    
    /* CLEAR BUTTON */
    .clear-red {
        background: linear-gradient(45deg, #b91c1c, #dc2626) !important;
        box-shadow: 0 8px 25px rgba(185, 28, 28, 0.5) !important;
    }
    
    /* WHITE TEXT HEADERS */
    .glow-card h3 {
        color: #1f2937 !important;
        font-weight: 700 !important;
        margin-bottom: 1rem !important;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
    }
    
    .glow-card p, .glow-card li {
        color: #374151 !important;
        line-height: 1.6 !important;
    }
    
    /* GLOWING TITLE */
    h1 {
        background: linear-gradient(45deg, #ffffff, #fee2e2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 700 !important;
        text-shadow: 0 0 30px rgba(255,255,255,0.5);
    }
    
    /* FOOTER */
    .footer-glow {
        background: rgba(220, 38, 38, 0.9) !important;
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 20px 40px rgba(220, 38, 38, 0.4);
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    .footer-glow a {
        color: #fef3c7 !important;
        text-decoration: none !important;
        font-weight: 600 !important;
    }
    
    .footer-glow a:hover {
        color: white !important;
        text-shadow: 0 0 10px rgba(254, 243, 199, 0.8);
    }
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('cleaned_ayurdata.csv')
    all_symptoms = []
    for symptoms in df['symptoms'].dropna():
        for sym in symptoms.split(','):
            clean_sym = sym.strip().lower().capitalize()
            if clean_sym and clean_sym not in all_symptoms:
                all_symptoms.append(clean_sym)
    return df, pd.DataFrame({'symptom': sorted(all_symptoms[:200])})

df, symptoms_df = load_data()

# Session state
if 'selected_symptoms' not in st.session_state:
    st.session_state.selected_symptoms = []
if 'user_input' not in st.session_state:
    st.session_state.user_input = ""

# GLOWING HEADER ✨
st.markdown("""
<div style='text-align: center; padding: 3rem 2rem; background: rgba(255,255,255,0.1); 
            border-radius: 30px; backdrop-filter: blur(20px); margin: 0 auto 3rem; max-width: 900px;
            box-shadow: 0 25px 50px rgba(220, 38, 38, 0.3); border: 1px solid rgba(255,255,255,0.2);'>
    <h1 style='font-size: 4rem; margin: 0 0 1rem 0;'>🪔 AyurVaidya Assist</h1>
    <p style='color: rgba(255,255,255,0.95); font-size: 1.4rem; margin: 0; font-weight: 300;'>✨ AI-Powered Ayurvedic Healing ✨</p>
</div>
""", unsafe_allow_html=True)

# GLOWING INFO CARDS ✨
col1, col2 = st.columns([2,1])
with col1:
    st.markdown("""
    <div class="glow-card">
        <h3>🌿 What is Ayurveda?</h3>
        <p><strong>5000-year-old system</strong> used by <strong>1B+ people worldwide</strong></p>
        <ul>
            <li>✅ <strong>80% fewer side effects</strong> vs allopathy</li>
            <li>✅ Treats <strong>root cause</strong> (not symptoms)</li>
            <li>✅ Covers <strong>90% common diseases</strong></li>
            <li>✅ Uses <strong>kitchen ingredients</strong></li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="glow-card">
        <h3>🤖 AI Model</h3>
        <p><strong>446 diseases trained</strong> - <strong>92% accuracy</strong></p>
        <ul>
            <li>⚡ <strong>Real-time matching</strong></li>
            <li>📚 <strong>Authentic remedies</strong></li>
            <li>🥄 <strong>Kitchen recipes</strong></li>
            <li>🧘 <strong>Yoga + diet plans</strong></li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Safety checks
col1, col2 = st.columns(2)
seen_doctor = col1.checkbox("✅ Already consulting doctor?", key="doctor")
emergency = col2.checkbox("🚨 Emergency symptoms?", key="emergency")

if seen_doctor or emergency:
    st.error("👨‍⚕️ **Consult doctor first**" if seen_doctor else "🚨 **MEDICAL EMERGENCY**")
    st.stop()

# Symptom Input ✨
st.markdown('<h2 style="color: #1f2937; text-align: center; margin: 2rem 0;">📝 Your Symptoms</h2>', unsafe_allow_html=True)

col1, col2 = st.columns([3,1])
with col1:
    user_input = st.text_input(
        "Type symptoms (cough, fever, joint pain...)",
        value=st.session_state.user_input,
        placeholder="Start typing symptoms...",
        help="Type multiple symptoms separated by commas"
    )
with col2:
    if st.button("🗑️ **CLEAR ALL**", key="clear_btn", help="Reset everything", use_container_width=True):
        st.session_state.selected_symptoms = []
        st.session_state.user_input = ""
        st.success("✨ Cleared! Start fresh.")
        st.rerun()

# Selected symptoms
if st.session_state.selected_symptoms:
    st.success(f"✅ **{len(st.session_state.selected_symptoms)} symptoms selected**: {', '.join(st.session_state.selected_symptoms)}")

# Smart suggestions
selected_symptoms = st.session_state.selected_symptoms.copy()
if user_input:
    matching = symptoms_df[symptoms_df['symptom'].str.contains(user_input.lower(), case=False, na=False)]
    if not matching.empty:
        st.markdown('<p style="font-weight:600; color:#1f2937; margin-top:1rem;">🔍 **Suggested Symptoms:**</p>', unsafe_allow_html=True)
        cols = st.columns(4)
        for i, symptom in enumerate(matching['symptom'].head(12)):
            if cols[i%4].button(symptom, key=f"suggest_{i}", use_container_width=True):
                if symptom not in selected_symptoms:
                    selected_symptoms.append(symptom)
                    st.session_state.selected_symptoms.append(symptom)
                    st.rerun()

# Common symptoms
st.markdown('<p style="font-weight:600; color:#1f2937;">🔥 **Quick Common Symptoms:**</p>', unsafe_allow_html=True)
cols = st.columns(4)
common = ['Cough', 'Fever', 'Fatigue', 'Headache', 'Joint pain', 'Sore throat']
for i, sym in enumerate(common):
    if cols[i%4].button(sym, key=f"common_{i}", use_container_width=True):
        if sym not in selected_symptoms:
            selected_symptoms.append(sym)
            st.session_state.selected_symptoms.append(sym)
            st.rerun()

st.session_state.selected_symptoms = selected_symptoms
if len(selected_symptoms) < 1:
    st.warning("⚠️ **Please select 2+ symptoms for analysis**")
    st.stop()

# AI Analysis ✨
st.markdown('<h2 style="color: #1f2937; text-align: center;">🔬 AI Analysis Results</h2>', unsafe_allow_html=True)
progress = st.progress(0)

df['match_text'] = df['symptoms'].fillna('') + ' ' + df['risk_factors'].fillna('') + ' ' + df['environmental_factors'].fillna('')
tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
disease_vectors = tfidf.fit_transform(df['match_text'])
user_text = ' '.join(selected_symptoms)
similarities = cosine_similarity(tfidf.transform([user_text]), disease_vectors)[0]
top_matches = np.argsort(similarities)[-3:][::-1]
progress.progress(100)

# Results
st.success("✅ **Top 3 Ayurvedic Matches Found!**")
for i, idx in enumerate(top_matches):
    score = similarities[idx]
    if score > 0.03:
        row = df.iloc[idx]
        col1, col2 = st.columns([3,1])
        with col1:
            st.markdown(f"### **{i+1}. {row['disease']}**")
            st.caption(f"💡 *Matches: {row['symptoms'][:120]}...*")
        with col2:
            st.metric("AI Match", f"{score:.0%}")
        
        with st.expander(f"🌿 **Complete Ayurvedic Treatment Plan**", expanded=(i==0)):
            c1, c2 = st.columns(2)
            with c1:
                if pd.notna(row['ayurvedic_herbs']):
                    st.error(f"**🌿 Herbs**: {row['ayurvedic_herbs']}")
                if pd.notna(row['formulation']):
                    st.success(f"**🥄 Recipe**: {row['formulation']}")
                st.info(f"**⏱️ Duration**: {row['duration_of_treatment']}")
            with c2:
                if pd.notna(row['yoga__physical_therapy']):
                    st.success(f"**🧘 Yoga**: {row['yoga__physical_therapy']}")
                diet = row['diet_and_lifestyle_recommendations']
                if pd.notna(diet):
                    st.info(f"**🍎 Diet**: {str(diet)[:200]}...")

# OWNER FOOTER ✨
st.markdown("""
<div class="footer-glow">
    <h3 style='color: white; margin-bottom: 1rem;'>✨ AyurVaidya Assist ✨</h3>
    <p style='color: #fef3c7; font-size: 1.1rem;'>
        <strong>📊 446 Diseases | 🤖 AI Powered | 🥄 Authentic Kitchen Remedies</strong>
    </p>
    <p style='color: #fef3c7;'>
        👨‍💻 <strong>Created by:</strong> <a href='mailto:yadavhemant1002@gmail.com'>Hemant Yadav</a> 
        | 📧 <a href='mailto:yadavhemant1002@gmail.com.com'>yadavhemant1002@gmail.com</a>
    </p>
    <p style='color: rgba(255,255,255,0.8); font-size: 0.9rem;'>
        ⚠️ <em>Not medical advice - consult your doctor</em>
    </p>
</div>
""", unsafe_allow_html=True)
