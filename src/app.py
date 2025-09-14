# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import datetime

# Set page config
st.set_page_config(
    page_title="Myco-Net: AI Fungal Network Interpreter",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    :root {
        --primary: #2E8B57; /* dark green */
        --primary-light: #3CB371; /* medium green */
        --secondary: #8B4513; /* saddle brown */
        --secondary-light: #A0522D; /* sienna */
        --accent: #FF7F50; /* coral */
        --background: #FFFFFF; /* white background for cards */
        --text: #222222; /* dark text */
        --warning-bg: #FFF8DC; /* cornsilk */
        --warning-border: #FFD700; /* gold */
        --danger-bg: #FFE4E1; /* misty rose */
        --danger-border: #FF4500; /* orange red */
        --success-bg: #E6F4EA; /* light green */
        --success-border: #32CD32; /* lime green */
        --info-bg: #E8F4F8; /* light blue */
        --info-border: #1E90FF; /* dodger blue */
    }
    
    body {
        color: var(--text);
        background-color: #F9FAFB;
    }
    .main-header {
        font-size: 3rem;
        color: var(--primary);
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 700;
    }
    .sub-header {
        font-size: 1.8rem;
        color: var(--primary-light);
        border-bottom: 3px solid var(--primary-light);
        padding-bottom: 0.5rem;
        margin-top: 2rem;
        font-weight: 600;
    }
    .metric-box {
        background-color: var(--background);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 6px solid var(--primary);
        margin-bottom: 1.5rem;
        box-shadow: 0 6px 10px rgba(0, 0, 0, 0.08);
        color: var(--text);
    }
    .prediction-box {
        background-color: var(--success-bg);
        padding: 20px;
        border-radius: 12px;
        border-left: 6px solid var(--success-border);
        margin: 1.5rem 0;
        box-shadow: 0 6px 10px rgba(0, 0, 0, 0.1);
        color: var(--text);
    }
    .critical {
        background-color: var(--danger-bg);
        border-left: 6px solid var(--danger-border);
        color: #660000;
    }
    .warning {
        background-color: var(--warning-bg);
        border-left: 6px solid var(--warning-border);
        color: #665500;
    }
    .feature-importance {
        background-color: var(--info-bg);
        padding: 20px;
        border-radius: 12px;
        margin: 1.5rem 0;
        box-shadow: 0 6px 10px rgba(0, 0, 0, 0.1);
        color: var(--text);
    }
    .sidebar .sidebar-content {
        background-color: #e6f2e6 !important;
        color: var(--text);
    }
    .stButton button {
        background-color: var(--primary);
        color: white;
        font-weight: 700;
        border-radius: 10px;
        padding: 0.6rem 1.2rem;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    .stButton button:hover {
        background-color: var(--primary-light);
        transform: translateY(-3px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.25);
    }
    .card {
        background-color: var(--background);
        border-radius: 12px;
        padding: 1.8rem;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1);
        margin-bottom: 2rem;
        border-left: 6px solid var(--primary);
        color: var(--text);
    }
    .fungal-card {
        border-left: 6px solid var(--secondary);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 52px;
        white-space: pre-wrap;
        background-color: #e6f2e6;
        border-radius: 10px 10px 0 0;
        gap: 10px;
        padding-top: 12px;
        padding-bottom: 12px;
        font-weight: 700;
        color: var(--text);
        box-shadow: inset 0 -3px 0 0 var(--primary-light);
        transition: background-color 0.3s ease;
    }
    .stTabs [aria-selected="true"] {
        background-color: var(--primary);
        color: white;
        box-shadow: none;
    }
    .history-item {
        background-color: var(--background);
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        border-left: 5px solid var(--primary-light);
    }
    .history-header {
        font-weight: 700;
        color: var(--primary);
        margin-bottom: 10px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    /* Scrollbar for sidebar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #e6f2e6;
    }
    ::-webkit-scrollbar-thumb {
        background-color: var(--primary);
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for storing results
if 'plant_results' not in st.session_state:
    st.session_state.plant_results = None
if 'fungal_results' not in st.session_state:
    st.session_state.fungal_results = None
if 'test_history' not in st.session_state:
    st.session_state.test_history = []

# --- Model and Encoder Loading ---
@st.cache_resource
def load_assets():
    try:
        plant_model = joblib.load('plant_health_model.pkl')
        myco_model = joblib.load('myco_net_model.pkl')
        le_species = joblib.load('species_encoder.pkl')
        le_light = joblib.load('light_encoder.pkl') 
        le_microbe = joblib.load('microbe_encoder.pkl')
        return plant_model, myco_model, le_species, le_light, le_microbe
    except FileNotFoundError:
        st.error("Model files not found! Please run train_and_save_models.py first.")
        st.stop()
plant_model, myco_model, le_species, le_light, le_microbe = load_assets()

# Title and description
st.markdown('<h1 class="main-header">🌿 Myco-Net: AI Fungal Network Interpreter</h1>', unsafe_allow_html=True)
st.markdown("### Revolutionizing Plant Health Monitoring through AI and Fungal Network Analysis")

# Sidebar navigation
st.sidebar.markdown("## 🌿 Navigation")
app_mode = st.sidebar.radio("Choose Mode", 
                           ["Home", "Plant Health Assessment", "Fungal Network Analysis", 
                            "Combined Results", "Test History", "Results Dashboard"],
                           index=0)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Quick Stats")
if st.session_state.plant_results:
    status = st.session_state.plant_results['status']
    if status == "Healthy":
        st.sidebar.success(f"🌱 Plant: {status}")
    elif status == "Moderate Stress":
        st.sidebar.warning(f"🌱 Plant: {status}")
    else:
        st.sidebar.error(f"🌱 Plant: {status}")

if st.session_state.fungal_results:
    risk = st.session_state.fungal_results['risk_level']
    if risk == "Low Risk":
        st.sidebar.success(f"🍄 Fungal: {risk}")
    else:
        st.sidebar.error(f"🍄 Fungal: {risk}")

st.sidebar.markdown(f"**Tests Conducted:** {len(st.session_state.test_history)}")

# --- Page Content ---
if app_mode == "Home":
    st.header("Welcome to Myco-Net!")
    st.markdown("""
    **Myco-Net** is an innovative AI system that interprets plant health through:
    - **Standard sensor data** for immediate stress detection
    - **Fungal network analysis** for early warning predictions
    
    ### How it Works:
    1. **Input** your plant's sensor data
    2. **Get instant analysis** of current health status
    3. **Receive early warnings** from fungal network data
    4. **View actionable recommendations** for farmers
    
    ### Key Features:
    - 🎯 **100% accuracy** in current stress detection (Plant Health Model)
    - 🔮 **98.13% accuracy** in early warning predictions (Myco-Net Model)
    - 🌱 **Sustainable** farming practices
    - 💰 **Cost-effective** for small-scale farmers
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image("https://images.unsplash.com/photo-1589923188937-cb64779f4abe?w=600", 
                 caption="AI-Powered Plant Health Monitoring")
    with col2:
        st.image("https://images.unsplash.com/photo-1598003807926-9a376a19e536?w=600", 
                 caption="Fungal Network Analysis")
    
    st.markdown("---")
    st.markdown("### 🚀 Get Started")
    st.info("Use the navigation menu on the left to begin your analysis. Start with **Plant Health Assessment** or **Fungal Network Analysis**.")

elif app_mode == "Plant Health Assessment":
    st.markdown('<h2 class="sub-header">🌱 Plant Health Assessment</h2>', unsafe_allow_html=True)
    
    # Use tabs for better organization
    tab1, tab2 = st.tabs(["📊 Input Parameters", "ℹ️ Information"])
    
    with tab1:
        ph_features = ['Soil_Moisture', 'Ambient_Temperature', 'Soil_Temperature',
                       'Humidity', 'Light_Intensity', 'Soil_pH',
                       'Nitrogen_Level', 'Phosphorus_Level', 'Potassium_Level',
                       'Chlorophyll_Content', 'Electrochemical_Signal']

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Environmental Sensors")
            soil_moisture = st.slider("Soil Moisture (%)", 0.0, 100.0, 45.0, help="Optimal range: 30-60%")
            ambient_temp = st.slider("Ambient Temperature (°C)", 0.0, 50.0, 25.0, help="Optimal range: 20-30°C")
            soil_temp = st.slider("Soil Temperature (°C)", 0.0, 50.0, 22.0, help="Optimal range: 18-24°C")
            humidity = st.slider("Humidity (%)", 0.0, 100.0, 60.0, help="Optimal range: 50-70%")
            light_intensity = st.slider("Light Intensity (lux)", 0.0, 100000.0, 50000.0, key="light_ph")
            
        with col2:
            st.markdown("#### Soil Composition")
            soil_ph = st.slider("Soil pH", 0.0, 14.0, 6.5, help="Optimal range: 6.0-7.0")
            nitrogen = st.slider("Nitrogen Level (ppm)", 0.0, 100.0, 45.0, help="Optimal range: 40-60 ppm")
            phosphorus = st.slider("Phosphorus Level (ppm)", 0.0, 100.0, 35.0, help="Optimal range: 30-50 ppm")
            potassium = st.slider("Potassium Level (ppm)", 0.0, 100.0, 40.0, help="Optimal range: 35-55 ppm")
            chlorophyll = st.slider("Chlorophyll Content", 0.0, 100.0, 65.0, key="chlorophyll_ph")
        
        st.markdown("#### Bio-signals")
        electrochemical = st.slider("Electrochemical Signal", -100.0, 100.0, 10.0, key="electro_ph")
        
        if st.button("Analyze Plant Health", type="primary", use_container_width=True):
            features = pd.DataFrame([[soil_moisture, ambient_temp, soil_temp, humidity, light_intensity,
                                        soil_ph, nitrogen, phosphorus, potassium, chlorophyll, electrochemical]],
                                      columns=ph_features)
            
            prediction = plant_model.predict(features)[0]
            probabilities = plant_model.predict_proba(features)[0]
            confidence = np.max(probabilities) * 100
            
            st.session_state.plant_results = {
                'status': prediction,
                'confidence': confidence,
                'soil_moisture': soil_moisture,
                'nitrogen': nitrogen,
                'soil_ph': soil_ph,
                'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            st.markdown("---")
            
            if prediction == "Healthy":
                st.markdown(f"""
                <div class="prediction-box">
                    <h2>✅ Health Status: {prediction}</h2>
                    <p><strong>Confidence:</strong> {confidence:.1f}%</p>
                    <p>Your plants are in good condition! Maintain current practices.</p>
                </div>
                """, unsafe_allow_html=True)
            elif prediction == "Moderate Stress":
                st.markdown(f"""
                <div class="prediction-box warning">
                    <h2>⚠️ Health Status: {prediction}</h2>
                    <p><strong>Confidence:</strong> {confidence:.1f}%</p>
                    <p>Recommendations:</p>
                    <ul>
                        <li>Increase irrigation (current: {soil_moisture}%)</li>
                        <li>Add nitrogen-rich fertilizer (current: {nitrogen} ppm)</li>
                        <li>Monitor soil pH levels (current: {soil_ph})</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="prediction-box critical">
                    <h2>🚨 Health Status: {prediction}</h2>
                    <p><strong>Confidence:</strong> {confidence:.1f}%</p>
                    <p>Immediate action required:</p>
                    <ul>
                        <li>Urgent irrigation needed (current: {soil_moisture}%)</li>
                        <li>Apply emergency fertilizer (current nitrogen: {nitrogen} ppm)</li>
                        <li>Check soil pH and adjust if needed (current: {soil_ph})</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("""
        ### About Plant Health Assessment
        
        This module analyzes standard plant health parameters to detect current stress conditions:
        
        - **Soil Moisture**: Water content in the soil
        - **Temperature**: Both ambient and soil temperatures
        - **Humidity**: Atmospheric moisture level
        - **Light Intensity**: Amount of light exposure
        - **Soil pH**: Acidity/alkalinity level
        - **Nutrients**: Nitrogen, Phosphorus, and Potassium levels
        - **Chlorophyll Content**: Indicator of photosynthetic activity
        - **Electrochemical Signals**: Plant's electrical response to environment
        
        The model uses a Random Forest classifier trained on extensive agricultural data to provide accurate health assessments.
        """)

elif app_mode == "Fungal Network Analysis":
    st.markdown('<h2 class="sub-header">🍄 Fungal Network Analysis</h2>', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["📊 Input Parameters", "ℹ️ Information"])
    
    with tab1:
        myco_features = ['Species_encoded', 'Light_encoded', 'Microbe_encoded', 'AMF', 
                         'PHN_Imp', 'NSC_Imp', 'LIG_Imp']

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Fungal Metrics & Tree Type")
            species = st.selectbox("Plant Species", le_species.classes_)
            amf_colonization = st.slider("AMF Colonization (%)", 0.0, 100.0, 65.0)
            nsc_imp = st.slider("NSC_Imp (Non-Structural Carbohydrate Proxy)", 0.0, 1.0, 0.5, step=0.01)
            lig_imp = st.slider("LIG_Imp (Lignin Content Proxy)", 0.0, 1.0, 0.5, step=0.01)
        with col2:
            st.markdown("#### Environmental Factors")
            light_level = st.selectbox("Light Condition", le_light.classes_)
            microbe_type = st.selectbox("Microbial Community", le_microbe.classes_)
            phn_imp = st.slider("PHN_Imp (Water Stress Proxy)", 0.0, 1.0, 0.5, step=0.01)
            
        if st.button("Analyze Fungal Network", type="primary", use_container_width=True):
            species_encoded = le_species.transform([species])[0]
            light_encoded = le_light.transform([light_level])[0]
            microbe_encoded = le_microbe.transform([microbe_type])[0]
            
            features = pd.DataFrame([[species_encoded, light_encoded, microbe_encoded, 
                                        amf_colonization, phn_imp, nsc_imp, lig_imp]],
                                      columns=myco_features)
            
            prediction = myco_model.predict(features)[0]
            probabilities = myco_model.predict_proba(features)[0]
            confidence = np.max(probabilities) * 100
            
            st.markdown("---")
            
            risk_level = "Low Risk" if prediction == 1 else "High Risk"
            
            st.session_state.fungal_results = {
                'risk_level': risk_level,
                'confidence': confidence,
                'amf_colonization': amf_colonization,
                'nsc_level': nsc_imp,
                'microbe_type': microbe_type,
                'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            if risk_level == "Low Risk":
                st.markdown(f"""
                <div class="prediction-box">
                    <h2>✅ Survival Prediction: LIKELY TO SURVIVE</h2>
                    <p><strong>Confidence:</strong> {confidence:.1f}%</p>
                    <p>The fungal network appears strong and supportive of plant health.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="prediction-box warning">
                    <h2>⚠️ Survival Prediction: AT RISK</h2>
                    <p><strong>Confidence:</strong> {confidence:.1f}%</p>
                    <p>The fungal network may be compromised. Immediate action is recommended.</p>
                </div>
                """, unsafe_allow_html=True)
                
            # Feature importance visualization
            st.markdown("#### Feature Importance")
            myco_importance = pd.DataFrame({'feature': myco_features, 'importance': myco_model.feature_importances_})
            myco_importance = myco_importance.sort_values('importance', ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='importance', y='feature', data=myco_importance, palette="Oranges_r", ax=ax)
            ax.set_xlabel('Importance')
            ax.set_title('Feature Importance in Fungal Network Analysis')
            st.pyplot(fig)
    
    with tab2:
        st.markdown("""
        ### About Fungal Network Analysis
        
        This module analyzes fungal network parameters to predict future plant health issues:
        
        - **Plant Species**: Different species have different fungal associations
        - **Light Conditions**: Affects fungal network development
        - **Microbial Community**: Composition of soil microbes
        - **AMF Colonization**: Arbuscular Mycorrhizal Fungi colonization percentage
        - **PHN_Imp**: Water stress proxy measurement
        - **NSC_Imp**: Non-Structural Carbohydrate proxy (energy reserves)
        - **LIG_Imp**: Lignin content proxy (structural integrity)
        
        The Myco-Net model uses these parameters to provide early warnings about potential plant health issues before they become visible.
        """)

elif app_mode == "Combined Results":
    st.header("📊 Combined Analysis Results")
    
    if st.session_state.plant_results and st.session_state.fungal_results:
        plant = st.session_state.plant_results
        fungal = st.session_state.fungal_results
        
        # Summary cards
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            if plant['status'] == "Healthy":
                st.success(f"🌱 **Plant Health: {plant['status']}**")
            elif plant['status'] == "Moderate Stress":
                st.warning(f"🌱 **Plant Health: {plant['status']}**")
            else:
                st.error(f"🌱 **Plant Health: {plant['status']}**")
            st.write(f"Confidence: {plant['confidence']:.1f}%")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col2:
            st.markdown('<div class="card fungal-card">', unsafe_allow_html=True)
            if fungal['risk_level'] == "Low Risk":
                st.success(f"🍄 **Fungal Network: {fungal['risk_level']}**")
            else:
                st.error(f"🍄 **Fungal Network: {fungal['risk_level']}**")
            st.write(f"Confidence: {fungal['confidence']:.1f}%")
            st.markdown('</div>', unsafe_allow_html=True)
                
        with col3:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            if plant['status'] == "Healthy" and fungal['risk_level'] == "Low Risk":
                st.success("✅ **Overall: Optimal Health**")
                st.write("Your plants are healthy now and likely to remain so.")
            elif plant['status'] == "High Stress" or fungal['risk_level'] == "High Risk":
                st.error("🚨 **Overall: Critical Attention Needed**")
                st.write("Immediate intervention is required.")
            else:
                st.warning("⚠️ **Overall: Monitoring Required**")
                st.write("Some parameters need attention.")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Detailed results
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 🌱 Plant Health Details")
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            st.write(f"**Status:** {plant['status']} ({plant['confidence']:.1f}% confidence)")
            st.write(f"**Soil Moisture:** {plant['soil_moisture']}%")
            st.write(f"**Nitrogen Level:** {plant['nitrogen']} ppm")
            st.write(f"**Soil pH:** {plant['soil_ph']}")
            st.write(f"**Last Updated:** {plant['timestamp']}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown("#### 🍄 Fungal Network Details")
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            st.write(f"**Risk Level:** {fungal['risk_level']} ({fungal['confidence']:.1f}% confidence)")
            st.write(f"**AMF Colonization:** {fungal['amf_colonization']}%")
            st.write(f"**NSC Level:** {fungal['nsc_level']}")
            st.write(f"**Microbial Community:** {fungal['microbe_type']}")
            st.write(f"**Last Updated:** {fungal['timestamp']}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Recommendations
        st.markdown("#### 💡 Recommendations")
        if plant['status'] == "Healthy" and fungal['risk_level'] == "Low Risk":
            st.success("""
            - Maintain current practices
            - Continue regular monitoring
            - No immediate action needed
            """)
        elif plant['status'] == "High Stress" or fungal['risk_level'] == "High Risk":
            st.error("""
            - Immediate intervention required
            - Adjust irrigation and fertilization
            - Consider soil amendments
            - Monitor closely for changes
            """)
        else:
            st.warning("""
            - Monitor specific parameters
            - Consider slight adjustments to practices
            - Schedule follow-up assessment
            """)
        
        if st.button("Save Combined Results to History", type="primary", use_container_width=True):
            test_entry = {
                'plant_results': plant,
                'fungal_results': fungal,
                'combined_timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            st.session_state.test_history.append(test_entry)
            st.success("Results saved to history!")
            st.rerun()

    else:
        st.warning("⚠️ Please run both Plant Health and Fungal Network analyses first!")
        col1, col2 = st.columns(2)
        with col1:
            if not st.session_state.plant_results:
                st.error("Plant Health Assessment not completed")
            else:
                st.success("Plant Health Assessment completed")
        with col2:
            if not st.session_state.fungal_results:
                st.error("Fungal Network Analysis not completed")
            else:
                st.success("Fungal Network Analysis completed")

elif app_mode == "Test History":
    st.header("📋 Test History")
    
    if st.session_state.test_history:
        st.info(f"Total tests conducted: {len(st.session_state.test_history)}")
        
        # Add filter options
        col1, col2 = st.columns(2)
        with col1:
            filter_status = st.selectbox("Filter by Plant Status", 
                                       ["All", "Healthy", "Moderate Stress", "High Stress"])
        with col2:
            filter_risk = st.selectbox("Filter by Fungal Risk", 
                                     ["All", "Low Risk", "High Risk"])
        
        # Filter tests based on selection
        filtered_tests = st.session_state.test_history.copy()
        
        if filter_status != "All":
            filtered_tests = [test for test in filtered_tests if test['plant_results']['status'] == filter_status]
        
        if filter_risk != "All":
            filtered_tests = [test for test in filtered_tests if test['fungal_results']['risk_level'] == filter_risk]
        
        if not filtered_tests:
            st.warning("No tests match the selected filters.")
        else:
            for i, test in enumerate(reversed(filtered_tests)):
                plant = test['plant_results']
                fungal = test['fungal_results']
                
                # Create a unique key for each expander
                expander_key = f"test_{i}_{test['combined_timestamp'].replace(' ', '_').replace(':', '-')}"
                
                with st.expander(f"Test on {test['combined_timestamp']}", expanded=False):
                    st.markdown(f"""
                    <div class="history-item">
                        <div class="history-header">
                            <span>Test from {test['combined_timestamp']}</span>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("🌱 Plant Health")
                        if plant['status'] == "Healthy":
                            st.success(f"**Status:** {plant['status']} ({plant['confidence']:.1f}%)")
                        elif plant['status'] == "Moderate Stress":
                            st.warning(f"**Status:** {plant['status']} ({plant['confidence']:.1f}%)")
                        else:
                            st.error(f"**Status:** {plant['status']} ({plant['confidence']:.1f}%)")
                        
                        st.write(f"Soil Moisture: {plant['soil_moisture']}%")
                        st.write(f"Nitrogen: {plant['nitrogen']} ppm")
                        st.write(f"Soil pH: {plant['soil_ph']}")
                    
                    with col2:
                        st.subheader("🍄 Fungal Network")
                        if fungal['risk_level'] == "Low Risk":
                            st.success(f"**Risk Level:** {fungal['risk_level']} ({fungal['confidence']:.1f}%)")
                        else:
                            st.error(f"**Risk Level:** {fungal['risk_level']} ({fungal['confidence']:.1f}%)")
                        
                        st.write(f"AMF Colonization: {fungal['amf_colonization']}%")
                        st.write(f"NSC Level: {fungal['nsc_level']}")
                        st.write(f"Microbe Type: {fungal['microbe_type']}")
                    
                    st.markdown("</div>", unsafe_allow_html=True)
        
        # Add export functionality
        st.markdown("---")
        st.subheader("Export History")
        
        if st.button("Export Test History as CSV", type="secondary"):
            # Create a DataFrame from the test history
            history_data = []
            for test in st.session_state.test_history:
                plant = test['plant_results']
                fungal = test['fungal_results']
                history_data.append({
                    'Timestamp': test['combined_timestamp'],
                    'Plant_Status': plant['status'],
                    'Plant_Confidence': plant['confidence'],
                    'Soil_Moisture': plant['soil_moisture'],
                    'Nitrogen_Level': plant['nitrogen'],
                    'Soil_pH': plant['soil_ph'],
                    'Fungal_Risk': fungal['risk_level'],
                    'Fungal_Confidence': fungal['confidence'],
                    'AMF_Colonization': fungal['amf_colonization'],
                    'NSC_Level': fungal['nsc_level'],
                    'Microbe_Type': fungal['microbe_type']
                })
            
            history_df = pd.DataFrame(history_data)
            csv = history_df.to_csv(index=False)
            
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="myco_net_test_history.csv",
                mime="text/csv",
            )
    else:
        st.info("📝 No test history yet. Run some analyses and save the results to build your history!")
        
    if st.session_state.test_history and st.button("🗑️ Clear All History", type="secondary"):
        st.session_state.test_history = []
        st.success("Test history cleared!")
        st.rerun()

elif app_mode == "Results Dashboard":
    st.header("📊 Performance Dashboard")
    
    ph_features = ['Soil_Moisture', 'Ambient_Temperature', 'Soil_Temperature',
                   'Humidity', 'Light_Intensity', 'Soil_pH',
                   'Nitrogen_Level', 'Phosphorus_Level', 'Potassium_Level',
                   'Chlorophyll_Content', 'Electrochemical_Signal']
    
    ph_importance_df = pd.DataFrame({'feature': ph_features, 'importance': plant_model.feature_importances_})
    
    myco_features_raw = ['Species', 'Light', 'Microbe', 'AMF', 'PHN_Imp', 'NSC_Imp', 'LIG_Imp']
    myco_importance_df = pd.DataFrame({'feature': myco_features_raw, 'importance': myco_model.feature_importances_})
    
    st.subheader("Feature Importance Comparison")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    ph_importance_df = ph_importance_df.sort_values('importance', ascending=False)
    bars1 = sns.barplot(x='importance', y='feature', data=ph_importance_df, ax=ax1, palette="Greens_r")
    ax1.set_xlabel('Importance')
    ax1.set_title('Plant Health Model Feature Importance')
    # Add value labels on bars
    for i, v in enumerate(ph_importance_df['importance']):
        ax1.text(v + 0.01, i, f'{v:.2f}', color='black', ha='left', va='center')
    
    myco_importance_df = myco_importance_df.sort_values('importance', ascending=False)
    bars2 = sns.barplot(x='importance', y='feature', data=myco_importance_df, ax=ax2, palette="Oranges_r")
    ax2.set_xlabel('Importance')
    ax2.set_title('Myco-Net Model Feature Importance')
    # Add value labels on bars
    for i, v in enumerate(myco_importance_df['importance']):
        ax2.text(v + 0.01, i, f'{v:.2f}', color='black', ha='left', va='center')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Model performance metrics
    st.subheader("Model Performance Metrics")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.metric("Plant Health Model Accuracy", "100%", "0%")
        st.write("Trained on extensive agricultural data")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card fungal-card">', unsafe_allow_html=True)
        st.metric("Myco-Net Model Accuracy", "98.13%", "1.87%")
        st.write("Early warning prediction system")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Add some statistics if we have test history
    if st.session_state.test_history:
        st.subheader("Historical Trends")
        
        # Create a DataFrame from the test history
        history_data = []
        for test in st.session_state.test_history:
            plant = test['plant_results']
            fungal = test['fungal_results']
            history_data.append({
                'Timestamp': test['combined_timestamp'],
                'Plant_Status': plant['status'],
                'Fungal_Risk': fungal['risk_level'],
                'Soil_Moisture': plant['soil_moisture'],
                'Nitrogen_Level': plant['nitrogen'],
                'AMF_Colonization': fungal['amf_colonization']
            })
        
        history_df = pd.DataFrame(history_data)
        history_df['Timestamp'] = pd.to_datetime(history_df['Timestamp'])
        history_df = history_df.sort_values('Timestamp')
        
        # Plot trends over time
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Soil moisture over time
        axes[0, 0].plot(history_df['Timestamp'], history_df['Soil_Moisture'], marker='o', color='blue')
        axes[0, 0].set_title('Soil Moisture Over Time')
        axes[0, 0].set_ylabel('Soil Moisture (%)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Nitrogen level over time
        axes[0, 1].plot(history_df['Timestamp'], history_df['Nitrogen_Level'], marker='o', color='green')
        axes[0, 1].set_title('Nitrogen Level Over Time')
        axes[0, 1].set_ylabel('Nitrogen (ppm)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # AMF colonization over time
        axes[1, 0].plot(history_df['Timestamp'], history_df['AMF_Colonization'], marker='o', color='orange')
        axes[1, 0].set_title('AMF Colonization Over Time')
        axes[1, 0].set_ylabel('AMF Colonization (%)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Status counts
        status_counts = history_df['Plant_Status'].value_counts()
        axes[1, 1].pie(status_counts.values, labels=status_counts.index, autopct='%1.1f%%', 
                      colors=['green', 'orange', 'red'])
        axes[1, 1].set_title('Plant Status Distribution')
        
        plt.tight_layout()
        st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px; font-size: 0.9rem;'>
    <p>Myco-Net: AI Fungal Network Interpreter | Developed for Sustainable Agriculture</p>
</div>
""", unsafe_allow_html=True)
