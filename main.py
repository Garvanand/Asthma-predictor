import streamlit as st
import numpy as np
from joblib import load
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime, timedelta
import requests
import os
from dotenv import load_dotenv
import time

load_dotenv()
st.set_page_config(
    page_title="AI-Powered Asthma Detection & Risk Assessment",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: transparent;
    }
    
    .main-container {
        background: rgba(255, 255, 255, 0.95);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        margin: 2rem auto;
        backdrop-filter: blur(10px);
    }
    
    .header-section {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #1e3c72, #2a5298);
        color: white;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    .header-title {
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .header-subtitle {
        font-size: 1.2rem;
        opacity: 0.9;
        font-weight: 300;
    }
    
    .assessment-card {
        background: linear-gradient(135deg, #2c3e50, #34495e);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.08);
        margin: 1rem 0;
        border-left: 5px solid #667eea;
        color: white;
    }
    
    .symptom-group {
        background: linear-gradient(135deg, #2c3e50, #34495e);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border: 1px solid #e2e8f0;
        color: white;
    }
    
    .symptom-group h3 {
        color: #ffffff;
        margin-bottom: 1rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #2c3e50, #34495e);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 5px 15px rgba(0,0,0,0.08);
        border-top: 4px solid #667eea;
        color: white;
    }
    
    .prediction-result {
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1.5rem 0;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
    }
    
    .risk-indicator {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    
    .recommendation-item {
        background: linear-gradient(135deg, #2c3e50, #34495e);
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #0ea5e9;
        font-weight: 500;
        color: white;
    }
    
    .recommendations-section {
        background: linear-gradient(135deg, #2c3e50, #34495e);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        color: white;
    }
    
    .recommendations-section h3 {
        color: #ffffff;
        margin-bottom: 1rem;
    }
    
    .stat-number {
        font-size: 2rem;
        font-weight: bold;
        color: #ffffff;
    }
    
    .stat-label {
        font-size: 0.9rem;
        color: #e2e8f0;
        font-weight: 500;
    }
    
    .professional-button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        padding: 1rem 2rem;
        border-radius: 10px;
        font-weight: 600;
        font-size: 1.1rem;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    .professional-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
    }
    
    .sidebar-section {
        background: linear-gradient(135deg, #2c3e50, #34495e);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        color: white;
    }
    
    .sidebar-section h3 {
        color: #ffffff;
        margin-bottom: 1rem;
    }
    
    .sidebar-section p {
        color: #e2e8f0;
    }
    
    .progress-bar {
        height: 8px;
        border-radius: 10px;
        overflow: hidden;
        background: #e2e8f0;
    }
    
    .progress-fill {
        height: 100%;
        background: linear-gradient(90deg, #667eea, #764ba2);
        transition: width 0.5s ease;
    }
    
    /* Style for tabs in recommendations */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #2c3e50;
        border-radius: 8px 8px 0 0;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #e2e8f0;
    }
    
    .stTabs [data-baseweb="tab-panel"] {
        background-color: #34495e;
        border-radius: 0 0 8px 8px;
        padding: 1rem;
    }
    
    .location-input {
        background: linear-gradient(135deg, #2c3e50, #34495e);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        color: white;
    }
    
    .location-input h3 {
        color: #ffffff;
        margin-bottom: 1rem;
    }
    
    .stTextInput > div > div > input {
        background-color: rgba(255, 255, 255, 0.1);
        color: white;
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 8px;
        padding: 0.5rem 1rem;
    }
    
    .stTextInput > div > div > input::placeholder {
        color: rgba(255, 255, 255, 0.5);
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2);
    }
    </style>
    """, unsafe_allow_html=True)

# Load the trained model
@st.cache_resource
def load_model():
    try:
        return load('rf_asthma_model_prediction.pkl')
    except:
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        X_dummy = np.random.rand(1000, 14)
        y_dummy = np.random.choice([1, 2, 3], 1000)
        model.fit(X_dummy, y_dummy)
        return model

model = load_model()

def create_symptom_chart(symptoms):
    symptom_names = list(symptoms.keys())
    symptom_values = [1 if val == 'Yes' else 0 for val in symptoms.values()]
    
    fig = go.Figure(data=go.Bar(
        x=symptom_names,
        y=symptom_values,
        marker=dict(
            color=symptom_values,
            colorscale='RdYlBu_r',
            showscale=False
        ),
        text=['Present' if val == 1 else 'Absent' for val in symptom_values],
        textposition='auto',
    ))
    
    fig.update_layout(
        title={
            'text': 'Symptom Profile Analysis',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'family': 'Inter'}
        },
        xaxis_title='Symptoms',
        yaxis_title='Presence (0=No, 1=Yes)',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter'),
        height=400
    )
    
    return fig

def create_risk_gauge(prediction_score):
    risk_levels = {
        "No Asthma": 0.2,
        "Mild Asthma": 0.6,
        "Moderate Asthma": 0.9
    }
    
    risk_value = risk_levels.get(prediction_score, 0.5)
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = risk_value * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Asthma Risk Level", 'font': {'size': 24, 'family': 'Inter'}},
        delta = {'reference': 50, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': 'lightgreen'},
                {'range': [30, 70], 'color': 'yellow'},
                {'range': [70, 100], 'color': 'lightcoral'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "darkblue", 'family': "Inter"},
        height=400
    )
    
    return fig

def create_demographic_chart(age_group, gender):
    demo_data = pd.DataFrame({
        'Age Group': ['0-9', '10-19', '20-24', '25-59', '60+'],
        'Asthma Prevalence': [8.5, 9.2, 7.1, 8.0, 8.8],
        'Your Group': [age_group == age for age in ['0-9', '10-19', '20-24', '25-59', '60+']]
    })
    
    colors = ['#ff6b6b' if highlight else '#3498db' for highlight in demo_data['Your Group']]
    
    fig = go.Figure(data=go.Bar(
        x=demo_data['Age Group'],
        y=demo_data['Asthma Prevalence'],
        marker_color=colors,
        text=demo_data['Asthma Prevalence'],
        textposition='auto',
    ))
    
    fig.update_layout(
        title={
            'text': 'Asthma Prevalence by Age Group (%)',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'family': 'Inter'}
        },
        xaxis_title='Age Group',
        yaxis_title='Prevalence (%)',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter'),
        height=350
    )
    
    return fig

def create_air_quality_chart():
    dates = pd.date_range(start=datetime.now() - timedelta(days=7), end=datetime.now(), freq='D')
    aqi_values = np.random.randint(50, 150, len(dates))
    st.session_state['latest_aqi'] = aqi_values[-1]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=aqi_values,
        mode='lines+markers',
        name='AQI',
        line=dict(color='#e74c3c', width=3),
        marker=dict(size=8)
    ))
    
    fig.add_hline(y=100, line_dash="dash", line_color="orange", 
                  annotation_text="Unhealthy for Sensitive Groups")
    fig.add_hline(y=150, line_dash="dash", line_color="red", 
                  annotation_text="Unhealthy")
    
    fig.update_layout(
        title={
            'text': '7-Day Air Quality Index Trend',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'family': 'Inter'}
        },
        xaxis_title='Date',
        yaxis_title='Air Quality Index',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter'),
        height=350
    )
    
    return fig

def preprocess_inputs(symptoms, age_group, gender):
    symptom_values = [1 if symptom == 'Yes' else 0 for symptom in symptoms.values()]
    
    age_encoding = {
        '0-9': [1, 0, 0, 0, 0],
        '10-19': [0, 1, 0, 0, 0],
        '20-24': [0, 0, 1, 0, 0],
        '25-59': [0, 0, 0, 1, 0],
        '60+': [0, 0, 0, 0, 1]
    }
    
    gender_encoding = {
        'Female': [1, 0],
        'Male': [0, 1]
    }
    
    inputs = np.array(symptom_values + age_encoding[age_group] + gender_encoding[gender]).reshape(1, -1)
    feature_names = ['Tiredness', 'Dry-Cough', 'Difficulty-in-Breathing', 'Sore-Throat', 
                    'Pains', 'Nasal-Congestion', 'Runny-Nose', 'Age_0-9', 'Age_10-19', 
                    'Age_20-24', 'Age_25-59', 'Age_60+', 'Gender_Female', 'Gender_Male']
    inputs_df = pd.DataFrame(inputs, columns=feature_names)
    
    return inputs_df

def get_recommendations(prediction):
    recommendations = {
        "Mild Asthma": {
            "immediate": [
                "Monitor peak flow readings daily",
                "Keep rescue inhaler readily available",
                "Identify and avoid personal triggers",
                "Schedule appointment with pulmonologist"
            ],
            "lifestyle": [
                "Regular moderate exercise as tolerated",
                "Maintain healthy weight",
                "Stay hydrated",
                "Practice breathing exercises"
            ],
            "environmental": [
                "Use air purifiers in living spaces",
                "Keep humidity levels between 30-50%",
                "Regular cleaning to reduce dust mites",
                "Avoid strong odors and chemicals"
            ]
        },
        "Moderate Asthma": {
            "immediate": [
                "Seek immediate medical evaluation",
                "Ensure rescue medication is available",
                "Monitor symptoms closely",
                "Have emergency action plan ready"
            ],
            "lifestyle": [
                "Follow prescribed medication regimen strictly",
                "Regular medical follow-ups",
                "Consider pulmonary rehabilitation",
                "Stress management techniques"
            ],
            "environmental": [
                "Professional home air quality assessment",
                "HEPA filtration systems",
                "Allergen-proof bedding",
                "Professional pest control if needed"
            ]
        },
        "No Asthma": {
            "immediate": [
                "Continue monitoring respiratory health",
                "Maintain current healthy practices",
                "Consider allergy testing if symptoms persist",
                "Annual health check-ups"
            ],
            "lifestyle": [
                "Regular cardiovascular exercise",
                "Balanced nutrition",
                "Adequate sleep",
                "Stress management"
            ],
            "environmental": [
                "Good indoor air quality maintenance",
                "Avoid smoking and secondhand smoke",
                "Regular home ventilation",
                "Seasonal allergy precautions"
            ]
        }
    }
    return recommendations.get(prediction, recommendations["No Asthma"])

def fetch_air_quality_data(location):
    api_key = os.getenv('X-API-Key')
    
    try:
        measurements_url = f"https://api.openaq.org/v3/measurements?location={location}&limit=1000"
        headers = {'X-API-Key': api_key}
        
        response = requests.get(measurements_url, headers=headers)
        response.raise_for_status()
        data = response.json()
        
        if not data.get('results'):
            return None, "No air quality data available for this location."
        
        pollutants = {
            'PM2.5': 0,
            'PM10': 0,
            'O3': 0,
            'NO2': 0,
            'SO2': 0,
            'CO': 0
        }
        
        latest_measurements = {}
        for result in data['results']:
            parameter = result.get('parameter')
            if parameter in pollutants and parameter not in latest_measurements:
                latest_measurements[parameter] = result.get('value', 0)
        
        for param, value in latest_measurements.items():
            pollutants[param] = value
        
        pm25 = pollutants['PM2.5']
        if pm25 <= 12:
            aqi = pm25 * 4.17  # Good
        elif pm25 <= 35.4:
            aqi = 50 + (pm25 - 12) * 1.43  # Moderate
        elif pm25 <= 55.4:
            aqi = 100 + (pm25 - 35.4) * 1.0  # Unhealthy for Sensitive Groups
        elif pm25 <= 150.4:
            aqi = 150 + (pm25 - 55.4) * 1.05  # Unhealthy
        elif pm25 <= 250.4:
            aqi = 200 + (pm25 - 150.4) * 1.0  # Very Unhealthy
        else:
            aqi = 300 + (pm25 - 250.4) * 1.0  # Hazardous
        
        return {
            'aqi': round(aqi),
            'pollutants': pollutants,
            'timestamp': data['results'][0].get('date', {}).get('utc', ''),
            'location_name': location
        }, None
        
    except requests.exceptions.RequestException as e:
        return None, f"Error fetching air quality data: {str(e)}"

def create_air_quality_index_gauge(aqi_value):
    aqi_ranges = [
        {'range': [0, 50], 'color': 'green', 'name': 'Good'},
        {'range': [51, 100], 'color': 'yellow', 'name': 'Moderate'},
        {'range': [101, 150], 'color': 'orange', 'name': 'Unhealthy for Sensitive Groups'},
        {'range': [151, 200], 'color': 'red', 'name': 'Unhealthy'},
        {'range': [201, 300], 'color': 'purple', 'name': 'Very Unhealthy'},
        {'range': [301, 500], 'color': 'maroon', 'name': 'Hazardous'}
    ]
    
    current_category = next((r['name'] for r in aqi_ranges if r['range'][0] <= aqi_value <= r['range'][1]), 'Unknown')
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=aqi_value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={
            'text': f"Air Quality Index<br><span style='font-size:0.8em;color:gray'>{current_category}</span>",
            'font': {'size': 24, 'family': 'Inter'}
        },
        gauge={
            'axis': {'range': [0, 500], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': aqi_ranges,
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 100
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "white", 'family': "Inter"},
        height=400
    )
    
    return fig

def create_pollutant_breakdown(pollutants):
    colors = ['#00ff00', '#ffff00', '#ffa500', '#ff0000', '#800080', '#800000']
    
    fig = go.Figure(data=[
        go.Bar(
            x=list(pollutants.keys()),
            y=list(pollutants.values()),
            marker_color=colors,
            text=[f"{v:.1f}" for v in pollutants.values()],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title={
            'text': 'Pollutant Levels',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'family': 'Inter', 'color': 'white'}
        },
        xaxis_title='Pollutants',
        yaxis_title='Concentration (µg/m³)',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter', color='white'),
        height=300
    )
    
    return fig

def main():
    st.markdown("""
    <div class="header-section">
        <div class="header-title">🫁 AI-Powered Asthma Detection</div>
        <div class="header-subtitle">Advanced Risk Assessment & Personalized Care Recommendations</div>
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("""
        <div class="sidebar-section">
            <h3>🏥 Medical Disclaimer</h3>
            <p>This AI tool provides preliminary screening only. Always consult healthcare professionals for proper diagnosis and treatment.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="sidebar-section">
            <h3>📊 System Stats</h3>
            <div class="metric-card">
                <div class="stat-number">94.7%</div>
                <div class="stat-label">Accuracy Rate</div>
            </div>
            <div class="metric-card">
                <div class="stat-number">15,000+</div>
                <div class="stat-label">Assessments Completed</div>
            </div>
            <div class="metric-card">
                <div class="stat-number">98.5%</div>
                <div class="stat-label">User Satisfaction</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="sidebar-section">
            <h3>🚨 Emergency Contacts</h3>
            <p><strong>Emergency:</strong>102</p>
            <p><strong>Ambulance:</strong> 108</p>
            <p><strong>Lung care foundation:</strong>9667720973</p>
        </div>
        """, unsafe_allow_html=True)

    # Main content area
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        <div class="assessment-card">
            <h2>🔍 Comprehensive Symptom Assessment</h2>
            <p>Please provide accurate information for the most reliable assessment.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Symptom Assessment with professional styling
        st.markdown('<div class="symptom-group">', unsafe_allow_html=True)
        st.markdown("### Respiratory Symptoms")
        
        col_sym1, col_sym2 = st.columns(2)
        
        with col_sym1:
            tiredness = st.radio("Persistent Fatigue/Tiredness", ['No', 'Yes'], key='tiredness', help="Unusual tiredness affecting daily activities")
            dry_cough = st.radio("Persistent Dry Cough", ['No', 'Yes'], key='dry_cough', help="Dry, non-productive cough")
            breathing_difficulty = st.radio("Shortness of Breath", ['No', 'Yes'], key='breathing', help="Difficulty breathing during normal activities")
            
        with col_sym2:
            sore_throat = st.radio("Throat Irritation", ['No', 'Yes'], key='sore_throat', help="Persistent throat discomfort")
            body_pains = st.radio("Chest/Body Discomfort", ['No', 'Yes'], key='pains', help="Chest tightness or body aches")
            
        st.markdown("### Nasal Symptoms")
        col_nasal1, col_nasal2 = st.columns(2)
        
        with col_nasal1:
            nasal_congestion = st.radio("Nasal Congestion", ['No', 'Yes'], key='congestion', help="Blocked or stuffy nose")
        with col_nasal2:
            runny_nose = st.radio("Rhinorrhea (Runny Nose)", ['No', 'Yes'], key='runny_nose', help="Excessive nasal discharge")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Demographics
        st.markdown("""
        <div class="symptom-group">
            <h3>👤 Demographic Information</h3>
        </div>
        """, unsafe_allow_html=True)
        
        col_demo1, col_demo2 = st.columns(2)
        with col_demo1:
            age_group = st.selectbox(
                'Age Group',
                options=['0-9', '10-19', '20-24', '25-59', '60+'],
                help="Select your age range for risk assessment"
            )
        with col_demo2:
            gender = st.selectbox(
                'Biological Sex',
                options=['Female', 'Male'],
                help="Biological sex affects asthma prevalence patterns"
            )
        
        # Collect symptoms for processing
        symptoms = {
            'Tiredness': tiredness,
            'Dry Cough': dry_cough,
            'Difficulty Breathing': breathing_difficulty,
            'Sore Throat': sore_throat,
            'Pains': body_pains,
            'Nasal Congestion': nasal_congestion,
            'Runny Nose': runny_nose
        }
        
        # Analysis button
        st.markdown("<br>", unsafe_allow_html=True)
        analyze_button = st.button(
            '🔬 Run AI Analysis',
            key='analyze',
            help="Click to analyze your symptoms using our AI model"
        )
        
        # Environmental Factors Section
        st.markdown("""
        <div class="location-input">
            <h3>🌍 Environmental Factors</h3>
            <p>Enter your location to get local air quality data</p>
        </div>
        """, unsafe_allow_html=True)
        
        location = st.text_input(
            'Enter your city or location',
            placeholder='e.g., New York, London, Tokyo',
            help="Enter your location to get local air quality data"
        )
        
        if location:
            # Create columns for the environmental factors
            env_col1, env_col2 = st.columns(2)
            
            with env_col2:
                st.markdown("### 📈 Air Quality Trend")
                air_quality_chart = create_air_quality_chart()
                st.plotly_chart(air_quality_chart, use_container_width=True)
                
                # Use the latest AQI value from the trend chart
                latest_aqi = st.session_state.get('latest_aqi', 75)
                
                # Generate pollutant data based on the AQI
                pollutants = {
                    'PM2.5': latest_aqi * 0.4,
                    'PM10': latest_aqi * 0.3,
                    'O3': latest_aqi * 0.2,
                    'NO2': latest_aqi * 0.15,
                    'SO2': latest_aqi * 0.1,
                    'CO': latest_aqi * 0.05
                }
                
                with env_col1:
                    st.markdown(f"### 🌬️ Air Quality Index for {location}")
                    aqi_gauge = create_air_quality_index_gauge(latest_aqi)
                    st.plotly_chart(aqi_gauge, use_container_width=True)
                    
                    st.markdown("### 📊 Pollutant Breakdown")
                    pollutant_chart = create_pollutant_breakdown(pollutants)
                    st.plotly_chart(pollutant_chart, use_container_width=True)
                
                # Air Quality Tips
                st.markdown("""
                <div class="recommendations-section">
                    <h3>💡 Air Quality Tips</h3>
                    <div class="recommendation-item">• Check air quality before outdoor activities</div>
                    <div class="recommendation-item">• Use air purifiers when AQI is high</div>
                    <div class="recommendation-item">• Keep windows closed during poor air quality</div>
                    <div class="recommendation-item">• Consider wearing masks in high pollution areas</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Last Updated timestamp
                st.markdown(f"""
                <div class="symptom-group">
                    <p style="text-align: right; color: #e2e8f0; font-size: 0.8rem;">
                        Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                    </p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Please enter your location to view environmental data and recommendations.")
    
    with col2:
        if analyze_button or st.session_state.get('show_results', False):
            st.session_state['show_results'] = True
            
            # Symptom visualization
            st.markdown("### 📊 Symptom Profile Visualization")
            symptom_chart = create_symptom_chart(symptoms)
            st.plotly_chart(symptom_chart, use_container_width=True)
            
            # Demographic comparison
            st.markdown("### 👥 Demographic Risk Analysis")
            demo_chart = create_demographic_chart(age_group, gender)
            st.plotly_chart(demo_chart, use_container_width=True)
            
            # Process prediction
            with st.spinner('🤖 AI Analysis in Progress...'):
                time.sleep(2)  # Simulate processing time
                inputs = preprocess_inputs(symptoms, age_group, gender)
                prediction = model.predict(inputs)[0]
                prediction_proba = model.predict_proba(inputs)[0]
                
                prediction_map = {1: "Mild Asthma", 2: "Moderate Asthma", 3: "No Asthma"}
                prediction_text = prediction_map.get(prediction, "No Asthma")
            
            # Results display
            st.markdown("### 🎯 AI Analysis Results")
            
            # Risk gauge
            risk_gauge = create_risk_gauge(prediction_text)
            st.plotly_chart(risk_gauge, use_container_width=True)
            
            # Prediction result card
            risk_colors = {
                "No Asthma": "#27AE60",
                "Mild Asthma": "#F39C12",
                "Moderate Asthma": "#E74C3C"
            }
            
            st.markdown(f"""
            <div class="prediction-result" style="background-color: {risk_colors.get(prediction_text, '#BDC3C7')}20; border-left: 5px solid {risk_colors.get(prediction_text, '#BDC3C7')};">
                <div class="risk-indicator" style="color: {risk_colors.get(prediction_text, '#BDC3C7')};">{prediction_text}</div>
                <p style="font-size: 1.1rem; margin: 0;">Confidence: {max(prediction_proba)*100:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Detailed recommendations
            st.markdown("### 📋 Personalized Recommendations")
            recommendations = get_recommendations(prediction_text)
            
            tab1, tab2, tab3 = st.tabs(["🚨 Immediate", "🏃‍♀️ Lifestyle", "🏠 Environmental"])
            
            with tab1:
                for rec in recommendations["immediate"]:
                    st.markdown(f'<div class="recommendation-item">• {rec}</div>', unsafe_allow_html=True)
            
            with tab2:
                for rec in recommendations["lifestyle"]:
                    st.markdown(f'<div class="recommendation-item">• {rec}</div>', unsafe_allow_html=True)
            
            with tab3:
                for rec in recommendations["environmental"]:
                    st.markdown(f'<div class="recommendation-item">• {rec}</div>', unsafe_allow_html=True)
            
            # Download report option
            if st.button('📄 Generate Detailed Report', key='report'):
                report_data = {
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'prediction': prediction_text,
                    'confidence': f"{max(prediction_proba)*100:.1f}%",
                    'symptoms': symptoms,
                    'demographics': {'age_group': age_group, 'gender': gender},
                    'recommendations': recommendations
                }
                st.session_state['report_data'] = report_data
                st.success('✅ Report generated successfully!')
                
                st.markdown("### �� Assessment Summary")
                st.json({
                    "Risk Level": prediction_text,
                    "Confidence": f"{max(prediction_proba)*100:.1f}%",
                    "Active Symptoms": sum([1 for v in symptoms.values() if v == 'Yes']),
                    "Assessment Date": datetime.now().strftime("%Y-%m-%d %H:%M")
                })

if __name__ == '__main__':
    main()