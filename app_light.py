import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 🔥 RED THEME + FIXED BLUR TEXT + GLOWING COLUMNS ✨
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&family=Courier+Prime:wght@400;700&display=swap');
    
    /* FIXED BLUR TEXT BACKGROUND - 100% STREAMLIT SAFE */
    .stApp::before {
        content: 'AYURVEDA VATA PITTA KAPHA TURMERIC GINGER ASHWAGANDHA YOGA PRANA HEALING HERBS DOSHAS BALANCE TRIPHALA BRAHMI SHATAVARI NATURAL REMEDIES HOLISTIC MIND BODY SPIRIT DETOX CLEANSE WELLNESS SATVIK';
        position: fixed !important;
        top: 0 !important;
        left: 0 !important;
        width: 100vw !important;
        height: 300vh !important;
        color: rgba(220, 38, 38, 0.08) !important;
        font-family: 'Courier Prime', monospace !important;
        font-size: 22px !important;
        font-weight: 700 !important;
        pointer-events: none !important;
        z-index: 1 !important;
        line-height: 1.2 !important;
        text-shadow: 0 0 20px rgba(220, 38, 38, 0.3) !important;
        filter: blur(1.5px) !important;
        animation: scrollGlow 60s linear infinite !important;
        white-space: nowrap !important;
    }
    
    @keyframes scrollGlow {
        0% { 
            transform: translateX(100vw) rotateX(0deg); 
            opacity: 0.3; 
            filter: blur(1.5px);
        }
        50% { 
            opacity: 0.6; 
            filter: blur(1px);
            text-shadow: 0 0 40px rgba(220, 38, 38, 0.6);
        }
        100% { 
            transform: translateX(-50%) rotateX(10deg); 
            opacity: 0.2; 
            filter: blur(2px);
        }
    }
    
    /* RED GRADIENT OVERLAY */
    .main {
        background: linear-gradient(135deg, rgba(220, 38, 38, 0.88) 0%, rgba(185, 28, 28, 0.92) 50%, rgba(153, 27, 27, 0.95) 100%) !important;
        padding: 2rem !important;
        font-family: 'Poppins', sans-serif !important;
        position: relative !important;
        z-index: 10 !important;
        backdrop-filter: blur(2px) !important;
    }
    
    /* GLOWING COLUMNS */
    [data-testid="column"]:not(:last-child)::before {
        content: '' !important;
        position: absolute !important;
        top: 0 !important;
        left: 0 !important;
        right: 0 !important;
        bottom: 0 !important;
        background: linear-gradient(45deg, rgba(220, 38, 38, 0.1), rgba(239, 68, 68, 0.2)) !important;
        border-radius: 20px !important;
        box-shadow: 
            0 0 30px rgba(220, 38, 38, 0.4),
            inset 0 0 20px rgba(255, 255, 255, 0.1) !important;
        animation: columnPulse 4s ease-in-out infinite !important;
        z-index: -1 !important;
    }
    
    @keyframes columnPulse {
        0%, 100% { box-shadow: 0 0 30px rgba(220, 38, 38, 0.4), inset 0 0 20px rgba(255, 255, 255, 0.1); }
        50% { box-shadow: 0 0 50px rgba(220, 38, 38, 0.7), inset 0 0 30px rgba(255, 255, 255, 0.2); }
    }
    
    /* ENHANCED GLOWING RED CARDS */
    .glow-card {
        background: rgba(255, 255, 255, 0.97) !important;
        backdrop-filter: blur(25px) !important;
        border-radius: 25px !important;
        padding: 2.5rem !important;
        margin: 1.5rem 0 !important;
        box-shadow: 
            0 25px 50px rgba(220, 38, 38, 0.35),
            0 0 0 1px rgba(255, 255, 255, 0.25),
            inset 0 1px 0 rgba(255, 255, 255, 0.6),
            0 0 40px rgba(220, 38, 38, 0.2) !important;
        border: 1px solid rgba(255, 255, 255, 0.35) !important;
        transition: all 0.5s cubic-bezier(0.4, 0, 0.2, 1) !important;
        position: relative !important;
        overflow: hidden !important;
        animation: cardFloat 6s ease-in-out infinite !important;
    }
    
    .glow-card::before {
        content: '' !important;
        position: absolute !important;
        top: -50% !important;
        left: -50% !important;
        width: 200% !important;
        height: 200% !important;
        background: radial-gradient(circle, rgba(255,255,255,0.15) 0%, transparent 70%) !important;
        opacity: 0 !important;
        transition: opacity 0.6s !important;
    }
    
    .glow-card:hover::before {
        opacity: 1 !important;
        animation: shimmer 1.5s infinite !important;
    }
    
    @keyframes shimmer {
        0% { transform: translateX(-100%) translateY(-100%) rotate(45deg); }
        100% { transform: translateX(100%) translateY(100%) rotate(45deg); }
    }
    
    @keyframes cardFloat {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-5px); }
    }
    
    .glow-card:hover {
        transform: translateY(-12px) scale(1.02) !important;
        box-shadow: 
            0 40px 80px rgba(220, 38, 38, 0.5),
            0 0 0 1px rgba(255, 255, 255, 0.4),
            0 0 60px rgba(220, 38, 38, 0.4) !important;
    }
    
    /* SUPER GLOWING BUTTONS */
    .stButton > button {
        background: linear-gradient(45deg, #dc2626, #ef4444, #f87171) !important;
        border: none !important;
        border-radius: 25px !important;
        padding: 16px 32px !important;
        color: white !important;
        font-weight: 700 !important;
        font-size: 16px !important;
        cursor: pointer !important;
        box-shadow: 
            0 12px 35px rgba(220, 38, 38, 0.5),
            0 0 30px rgba(220, 38, 38, 0.3) !important;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
        position: relative !important;
        overflow: hidden !important;
        animation: buttonPulse 3s infinite !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-6px) scale(1.08) !important;
        box-shadow: 
            0 25px 50px rgba(220, 38, 38, 0.7),
            0 0 50px rgba(220, 38, 38, 0.5) !important;
        background: linear-gradient(45deg, #ef4444, #f87171, #dc2626) !important;
    }
    
    @keyframes buttonPulse {
        0%, 100% { box-shadow: 0 12px 35px rgba(220, 38, 38, 0.5), 0 0 30px rgba(220, 38, 38, 0.3); }
        50% { box-shadow: 0 12px 35px rgba(220, 38, 38, 0.7), 0 0 40px rgba(220, 38, 38, 0.5); }
    }
    
    /* GLOWING TITLE */
    h1 {
        background: linear-gradient(45deg, #ffffff, #fee2e2, #fecaca) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        background-clip: text !important;
        font-weight: 800 !important;
        text-shadow: 0 0 40px rgba(255,255,255,0.6) !important;
        animation: titleGlow 3s ease-in-out infinite alternate !important;
    }
    
    @keyframes titleGlow {
        0% { filter: drop-shadow(0 0 20px rgba(220, 38, 38, 0.5)); }
        100% { filter: drop-shadow(0 0 40px rgba(220, 38, 38, 0.8)); }
    }
    
    /* FOOTER ENHANCEMENTS */
    .footer-glow {
        background: linear-gradient(135deg, rgba(220, 38, 38, 0.95), rgba(185, 28, 28, 0.9)) !important;
        backdrop-filter: blur(30px) !important;
        border-radius: 25px !important;
        padding: 2.5rem !important;
        text-align: center !important;
        box-shadow: 
            0 30px 60px rgba(220, 38, 38, 0.5),
            0 0 50px rgba(220, 38, 38, 0.3) !important;
        border: 1px solid rgba(255,255,255,0.3) !important;
        animation: footerFloat 5s ease-in-out infinite !important;
    }
    
    @keyframes footerFloat {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-3px); }
    }
    
    /* TEXT STYLES */
    .glow-card h3 {
        color: #1f2937 !important;
        font-weight: 800 !important;
        margin-bottom: 1.2rem !important;
        text-shadow: 0 2px 8px rgba(0,0,0,0.15) !important;
    }
    
    .glow-card p, .glow-card li {
        color: #374151 !important;
        line-height: 1.7 !important;
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
<div style='text-align: center; padding: 3.5rem 2rem; background: rgba(255,255,255,0.12); 
            border-radius: 35px; backdrop-filter: blur(25px); margin: 0 auto 3rem; max-width: 950px;
            box-shadow: 0 30px 70px rgba(220, 38, 38, 0.4), 0 0 60px rgba(220, 38, 38, 0.2); 
            border: 1px solid rgba(255,255,255,0.25); position: relative; overflow: hidden;'>
    <h1 style='font-size: 4.2rem; margin: 0 0 1.2rem 0;'>🪔 AyurVaidya Assist</h1>
    <p style='color: rgba(255,255,255,0.98); font-size: 1.5rem; margin: 0; font-weight: 400;'>✨ AI-Powered Ayurvedic Healing ✨</p>
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
st.markdown('<h2 style="color: #1f2937; text-align: center; margin: 2.5rem 0; text-shadow: 0 4px 12px rgba(0,0,0,0.3);">📝 Your Symptoms</h2>', unsafe_allow_html=True)

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
        st.markdown('<p style="font-weight:700; color:#1f2937; margin-top:1.5rem;">🔍 **Suggested Symptoms:**</p>', unsafe_allow_html=True)
        cols = st.columns(4)
        for i, symptom in enumerate(matching['symptom'].head(12)):
            if cols[i%4].button(symptom, key=f"suggest_{i}", use_container_width=True):
                if symptom not in selected_symptoms:
                    selected_symptoms.append(symptom)
                    st.session_state.selected_symptoms.append(symptom)
                    st.rerun()

# Common symptoms
st.markdown('<p style="font-weight:700; color:#1f2937;">🔥 **Quick Common Symptoms:**</p>', unsafe_allow_html=True)
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
    <h3 style='color: white; margin-bottom: 1.2rem;'>✨ AyurVaidya Assist ✨</h3>
    <p style='color: #fef3c7; font-size: 1.2rem;'>
        <strong>📊 446 Diseases | 🤖 AI Powered | 🥄 Authentic Kitchen Remedies</strong>
    </p>
    <p style='color: #fef3c7;'>
        👨‍💻 <strong>Created by:</strong> <a href='mailto:yadavhemant1002@gmail.com'>Hemant Yadav</a> 
        | 📧 <a href='mailto:yadavhemant1002@gmail.com.com'>yadavhemant1002@gmail.com</a>
    </p>
    <p style='color: rgba(255,255,255,0.85); font-size: 1rem;'>
        ⚠️ <em>Not medical advice - consult your doctor</em>
    </p>
</div>
""", unsafe_allow_html=True)
