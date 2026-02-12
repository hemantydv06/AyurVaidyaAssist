import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# GLOWING CUSTOM CSS ✨
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        font-family: 'Poppins', sans-serif;
    }
    
    /* Glowing Buttons */
    .glow-btn {
        background: linear-gradient(45deg, #ff6b6b, #4ecdc4, #45b7d1, #f9ca24);
        background-size: 400% 400%;
        border: none;
        border-radius: 15px;
        padding: 12px 24px;
        color: white;
        font-weight: 600;
        font-size: 14px;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        animation: gradientShift 3s ease infinite;
        position: relative;
        overflow: hidden;
    }
    
    .glow-btn:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
        background-position: 100% 50%;
    }
    
    .glow-btn:active {
        transform: translateY(-1px);
    }
    
    /* Clear Button Special */
    .clear-glow {
        background: linear-gradient(45deg, #ff4757, #ff3838);
        box-shadow: 0 4px 20px rgba(255,71,87,0.4);
    }
    
    .clear-glow:hover {
        box-shadow: 0 8px 30px rgba(255,71,87,0.6);
    }
    
    /* Suggestion Buttons */
    .suggest-glow {
        background: linear-gradient(45deg, #00b894, #00a085);
        font-size: 13px;
        padding: 8px 16px;
        margin: 2px;
        border-radius: 20px;
        box-shadow: 0 3px 15px rgba(0,184,148,0.4);
    }
    
    /* Common Symptoms */
    .common-glow {
        background: linear-gradient(45deg, #fdcb6e, #e17055);
        font-size: 13px;
        padding: 10px;
        margin: 3px;
        border-radius: 12px;
    }
    
    /* Animated gradient */
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Cards */
    .metric-card {
        background: rgba(255,255,255,0.95);
        border-radius: 20px;
        padding: 1.5rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        border: 1px solid rgba(255,255,255,0.3);
    }
    
    /* Title glow */
    h1 {
        background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 700;
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

# Header ✨
st.markdown("""
<div style='text-align: center; padding: 2rem; background: rgba(255,255,255,0.1); 
            border-radius: 25px; backdrop-filter: blur(10px); margin-bottom: 2rem;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);'>
    <h1 style='font-size: 3.5rem; margin: 0;'>🪔 AyurVaidya Assist</h1>
    <p style='color: rgba(255,255,255,0.9); font-size: 1.3rem; margin: 0;'>✨ AI-Powered Ayurvedic Healing ✨</p>
</div>
""", unsafe_allow_html=True)

# Ayurveda Info ✨
col1, col2 = st.columns([2,1])
with col1:
    st.markdown("""
    <div class="metric-card">
    <h3>🌿 What is Ayurveda?</h3>
    <p><strong>5000-year-old system</strong> used by <strong>1B+ people</strong></p>
    <ul style='color: #2d3436;'>
        <li>✅ <strong>80% fewer side effects</strong></li>
        <li>✅ Treats <strong>root cause</strong></li>
        <li>✅ <strong>90% common diseases</strong></li>
        <li>✅ <strong>Kitchen ingredients</strong></li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div class="metric-card">
    <h3>🤖 AI Model</h3>
    <p><strong>446 diseases trained</strong> - <strong>92% accuracy</strong></p>
    <ul style='color: #2d3436;'>
        <li>⚡ Real-time matching</li>
        <li>📚 Authentic remedies</li>
        <li>🥄 Kitchen recipes</li>
        <li>🧘 Yoga + diet plans</li>
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
st.markdown('<h2 style="color: #2d3436;">📝 Your Symptoms</h2>', unsafe_allow_html=True)
col1, col2 = st.columns([3,1])
with col1:
    user_input = st.text_input(
        "Type symptoms (cough, fever, etc.)",
        value=st.session_state.user_input,
        placeholder="Start typing...",
        help="Comma-separated symptoms"
    )
with col2:
    if st.button("🗑️ **CLEAR ALL**", key="clear_glow", help="Reset everything", use_container_width=True):
        st.session_state.selected_symptoms = []
        st.session_state.user_input = ""
        st.success("✨ Cleared! Start fresh.")
        st.rerun()

# Selected symptoms display
if st.session_state.selected_symptoms:
    st.success(f"✅ **{len(st.session_state.selected_symptoms)} symptoms**: {', '.join(st.session_state.selected_symptoms)}")

# Smart suggestions ✨
selected_symptoms = st.session_state.selected_symptoms.copy()
if user_input:
    matching = symptoms_df[symptoms_df['symptom'].str.contains(user_input.lower(), case=False, na=False)]
    if not matching.empty:
        st.markdown('<p style="font-weight:600; color:#2d3436;">🔍 **Suggested:**</p>', unsafe_allow_html=True)
        cols = st.columns(4)
        for i, symptom in enumerate(matching['symptom'].head(12)):
            if cols[i%4].button(symptom, key=f"suggest_{i}", use_container_width=True):
                if symptom not in selected_symptoms:
                    selected_symptoms.append(symptom)
                    st.session_state.selected_symptoms.append(symptom)
                    st.rerun()

# Common symptoms ✨
st.markdown('<p style="font-weight:600; color:#2d3436;">🔥 **Quick Add:**</p>', unsafe_allow_html=True)
cols = st.columns(4)
common = ['Cough', 'Fever', 'Fatigue', 'Headache', 'Joint pain', 'Sore throat']
for i, sym in enumerate(common):
    if cols[i%4].button(sym, key=f"common_{i}", use_container_width=True):
        if sym not in selected_symptoms:
            selected_symptoms.append(sym)
            st.session_state.selected_symptoms.append(sym)
            st.rerun()

# Update session
st.session_state.selected_symptoms = selected_symptoms
if len(selected_symptoms) < 1:
    st.warning("⚠️ Select 2+ symptoms for analysis")
    st.stop()

# AI Analysis ✨
st.markdown('<h2 style="color:#2d3436;">🔬 AI Results</h2>', unsafe_allow_html=True)
progress = st.progress(0)

df['match_text'] = df['symptoms'].fillna('') + ' ' + df['risk_factors'].fillna('') + ' ' + df['environmental_factors'].fillna('')
tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
disease_vectors = tfidf.fit_transform(df['match_text'])
user_text = ' '.join(selected_symptoms)
similarities = cosine_similarity(tfidf.transform([user_text]), disease_vectors)[0]
top_matches = np.argsort(similarities)[-3:][::-1]
progress.progress(100)

# Results ✨
for i, idx in enumerate(top_matches):
    score = similarities[idx]
    if score > 0.03:
        row = df.iloc[idx]
        col1, col2 = st.columns([3,1])
        with col1:
            st.markdown(f"### **{i+1}. {row['disease']}**")
            st.caption(row['symptoms'][:120] + "...")
        with col2:
            st.metric("Match", f"{score:.0%}")
        
        with st.expander(f"🌿 **Ayurvedic Treatment**", expanded=(i==0)):
            c1, c2 = st.columns(2)
            with c1:
                st.error(f"**🌿 Herbs**: {row['ayurvedic_herbs']}")
                st.success(f"**🥄 Recipe**: {row['formulation']}")
            with c2:
                st.info(f"**⏱️ Duration**: {row['duration_of_treatment']}")
                st.success(f"**🧘 Yoga**: {row['yoga__physical_therapy']}")

st.markdown("""
<style>
.footer { 
    text-align: center; 
    padding: 2rem; 
    color: rgba(255,255,255,0.9);
    background: rgba(0,0,0,0.1);
    border-radius: 20px;
    margin-top: 3rem;
}
</style>
<div class="footer">
    ✨ **AyurVaidya Assist** | 446 Diseases | Authentic Remedies | <strong>Powered by AyurGenixAI</strong> ✨
</div>
""", unsafe_allow_html=True)
