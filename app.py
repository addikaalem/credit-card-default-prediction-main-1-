import streamlit as st
import pandas as pd
import pickle
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# ==================== PAGE CONFIGURATION ====================
st.set_page_config(
    page_title="Credit Card Default Predictor",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS FOR PROFESSIONAL UI ====================
st.markdown("""
<style>
    /* Main container */
    .main {
        padding: 0rem 1rem;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Result cards */
    .result-card {
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        margin: 1rem 0;
        animation: slideIn 0.5s ease;
    }
    
    @keyframes slideIn {
        from { transform: translateY(-20px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }
    
    .default-card {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        color: white;
    }
    
    .no-default-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        padding: 1.2rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        transition: transform 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    /* Risk factor items */
    .risk-item {
        padding: 0.8rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        background: #f8f9fa;
        border-left: 4px solid;
    }
    
    .risk-high { border-left-color: #dc2626; background: #fee2e2; }
    .risk-medium { border-left-color: #f59e0b; background: #fef3c7; }
    .risk-low { border-left-color: #10b981; background: #d1fae5; }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1.2rem;
        font-weight: 600;
        border-radius: 12px;
        width: 100%;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.2);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background-color: #f8f9fa;
        padding: 0.5rem;
        border-radius: 12px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
    
    /* Info box */
    .info-box {
        background: #f0fdf4;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #10b981;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: #fffbeb;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #f59e0b;
        margin: 1rem 0;
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #10b981, #f59e0b, #dc2626);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #666;
        font-size: 0.8rem;
        border-top: 1px solid #e5e7eb;
        margin-top: 2rem;
    }
    
    /* Data table styling */
    .dataframe {
        font-size: 0.9rem;
    }
    
    .dataframe th {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.75rem;
    }
</style>
""", unsafe_allow_html=True)

# ==================== LOAD MODEL ====================
@st.cache_resource
def load_model():
    """Load the trained logistic regression model"""
    try:
        with open('LogR_Model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error("❌ Model file 'LogR_Model.pkl' not found!")
        st.info("Please make sure LogR_Model.pkl is in the same folder")
        return None
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

model = load_model()

# If model not loaded, stop
if model is None:
    st.stop()

# ==================== HELPER FUNCTIONS ====================
def calculate_risk_metrics(pay_values, bill_amts, pay_amts, limit_bal, age, education, marriage):
    """Calculate various risk metrics"""
    
    late_payments = sum(1 for p in pay_values if p > 0)
    avg_bill = np.mean(bill_amts) if bill_amts else 0
    avg_pay = np.mean(pay_amts) if pay_amts else 0
    pay_ratio = avg_pay / avg_bill if avg_bill > 0 else 0
    utilization = avg_bill / limit_bal if limit_bal > 0 else 1
    
    # Calculate risk score
    risk_score = 0
    
    # Payment history (40% weight)
    if late_payments == 0:
        risk_score += 0
    elif late_payments <= 2:
        risk_score += 20
    else:
        risk_score += 40
    
    # Utilization (20% weight)
    if utilization < 0.3:
        risk_score += 0
    elif utilization <= 0.6:
        risk_score += 10
    else:
        risk_score += 20
    
    # Payment ratio (20% weight)
    if pay_ratio > 0.8:
        risk_score += 0
    elif pay_ratio >= 0.4:
        risk_score += 10
    else:
        risk_score += 20
    
    # Age (10% weight)
    if 35 <= age <= 60:
        risk_score += 0
    elif 25 <= age <= 34 or 61 <= age <= 70:
        risk_score += 5
    else:
        risk_score += 10
    
    # Education (5% weight)
    if education == "Graduate School":
        risk_score += 0
    elif education == "University":
        risk_score += 2
    else:
        risk_score += 5
    
    # Marriage (5% weight)
    if marriage == "Married":
        risk_score += 0
    elif marriage == "Single":
        risk_score += 3
    else:
        risk_score += 5
    
    return {
        'late_payments': late_payments,
        'avg_bill': avg_bill,
        'avg_pay': avg_pay,
        'pay_ratio': pay_ratio,
        'utilization': utilization,
        'risk_score': risk_score
    }

def get_risk_factors(metrics, age, education, marriage, limit_bal):
    """Get list of risk factors"""
    factors = []
    
    if metrics['late_payments'] == 0:
        factors.append(('low', '✅ No late payments', 'Excellent payment history'))
    elif metrics['late_payments'] <= 2:
        factors.append(('medium', f'⚠️ {metrics["late_payments"]} late payment(s)', 'Moderate payment risk'))
    else:
        factors.append(('high', f'🔴 {metrics["late_payments"]} late payments', 'High payment risk - critical'))
    
    if metrics['utilization'] < 0.3:
        factors.append(('low', '✅ Low credit utilization', f'{metrics["utilization"]:.0%} of limit used'))
    elif metrics['utilization'] <= 0.6:
        factors.append(('medium', f'⚠️ Moderate utilization', f'{metrics["utilization"]:.0%} of limit used'))
    else:
        factors.append(('high', f'🔴 High utilization', f'{metrics["utilization"]:.0%} of limit used - overextended'))
    
    if metrics['pay_ratio'] > 0.8:
        factors.append(('low', '✅ Full payments', f'Paying {metrics["pay_ratio"]:.0%} of bills'))
    elif metrics['pay_ratio'] >= 0.4:
        factors.append(('medium', f'⚠️ Partial payments', f'Paying only {metrics["pay_ratio"]:.0%} of bills'))
    else:
        factors.append(('high', f'🔴 Minimum payments', f'Paying only {metrics["pay_ratio"]:.0%} of bills - high risk'))
    
    if 35 <= age <= 60:
        factors.append(('low', '✅ Prime age', f'Age {age} - stable demographic'))
    elif 25 <= age <= 34 or 61 <= age <= 70:
        factors.append(('medium', f'⚠️ Age {age}', 'Moderate risk age group'))
    else:
        factors.append(('high', f'🔴 Age {age}', 'High risk age group'))
    
    if education == "Graduate School":
        factors.append(('low', '✅ Graduate education', 'Lower default rate historically'))
    elif education == "University":
        factors.append(('medium', f'📊 {education}', 'Average default rate'))
    else:
        factors.append(('medium', f'📊 {education}', 'Higher default rate historically'))
    
    if limit_bal > 300000:
        factors.append(('low', '✅ High credit limit', f'NT$ {limit_bal:,} - established credit'))
    elif limit_bal >= 100000:
        factors.append(('medium', f'💳 Medium credit limit', f'NT$ {limit_bal:,}'))
    else:
        factors.append(('medium', f'💳 Low credit limit', f'NT$ {limit_bal:,} - new or limited history'))
    
    return factors

def predict_single_customer(features, edu_map, marriage_map):
    """Predict for a single customer"""
    probability = model.predict_proba(features)[0][1]
    prediction_class = 1 if probability >= 0.5 else 0
    return probability, prediction_class

def predict_batch(df, edu_map, marriage_map):
    """Predict for batch of customers"""
    # Prepare all features
    features = df[['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
                   'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
                   'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
                   'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']].values
    
    probabilities = model.predict_proba(features)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    
    return probabilities, predictions

def display_prediction_results(probability, prediction_class, metrics, risk_factors, customer_info=None):
    """Display prediction results for a single customer"""
    
    # Main result card
    if prediction_class == 1:
        st.markdown(f"""
        <div class="result-card default-card">
            <h1>⚠️ DEFAULT PREDICTED</h1>
            <h2>This customer is LIKELY TO DEFAULT</h2>
            <p style="font-size: 1.5rem;">Default Probability: {probability:.1%}</p>
            <p>Risk Score: {metrics['risk_score']}/100</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.error("🚨 **HIGH RISK ALERT:** This customer has a high probability of defaulting on their credit card payment. Strongly recommend DECLINING the application or requiring significant collateral.")
        
    else:
        st.markdown(f"""
        <div class="result-card no-default-card">
            <h1>✅ NO DEFAULT PREDICTED</h1>
            <h2>This customer is UNLIKELY TO DEFAULT</h2>
            <p style="font-size: 1.5rem;">Default Probability: {probability:.1%}</p>
            <p>Risk Score: {metrics['risk_score']}/100</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.success("✅ **LOW RISK:** This customer appears creditworthy. Recommend APPROVING the application.")
    
    # Risk Metrics Dashboard
    st.markdown("### 📊 Risk Metrics Dashboard")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>📅 Late Payments</h4>
            <h2>{metrics['late_payments']}/6</h2>
            <small>months</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4>📈 Utilization</h4>
            <h2>{metrics['utilization']:.0%}</h2>
            <small>of limit</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h4>💸 Payment Ratio</h4>
            <h2>{metrics['pay_ratio']:.0%}</h2>
            <small>paid vs billed</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h4>💰 Avg Bill</h4>
            <h2>NT$ {metrics['avg_bill']:,.0f}</h2>
            <small>monthly</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown(f"""
        <div class="metric-card">
            <h4>🎯 Risk Score</h4>
            <h2>{metrics['risk_score']}</h2>
            <small>/100</small>
        </div>
        """, unsafe_allow_html=True)
    
    # Probability Gauge
    st.markdown("### 🎯 Default Probability Gauge")
    
    if probability < 0.18:
        gauge_color = "#10b981"
    elif probability < 0.40:
        gauge_color = "#f59e0b"
    elif probability < 0.65:
        gauge_color = "#f97316"
    else:
        gauge_color = "#dc2626"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        delta={"reference": 50, "increasing": {"color": "red"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#cbd5e1"},
            "bar": {"color": gauge_color},
            "bgcolor": "white",
            "borderwidth": 2,
            "bordercolor": "#e5e7eb",
            "steps": [
                {"range": [0, 18], "color": "#d1fae5"},
                {"range": [18, 40], "color": "#fef3c7"},
                {"range": [40, 65], "color": "#fed7aa"},
                {"range": [65, 100], "color": "#fee2e2"}
            ],
            "threshold": {
                "line": {"color": "red", "width": 4},
                "thickness": 0.75,
                "value": probability * 100
            }
        },
        title={"text": "Default Risk Probability", "font": {"size": 24}}
    ))
    fig.update_layout(height=350, margin=dict(l=20, r=20, t=80, b=20))
    st.plotly_chart(fig, use_container_width=True)
    
    # Risk Factors Analysis
    st.markdown("### ⚠️ Risk Factors Analysis")
    
    for level, factor, detail in risk_factors:
        if level == 'high':
            st.markdown(f"""
            <div class="risk-item risk-high">
                <strong>{factor}</strong><br>
                <small>{detail}</small>
            </div>
            """, unsafe_allow_html=True)
        elif level == 'medium':
            st.markdown(f"""
            <div class="risk-item risk-medium">
                <strong>{factor}</strong><br>
                <small>{detail}</small>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="risk-item risk-low">
                <strong>{factor}</strong><br>
                <small>{detail}</small>
            </div>
            """, unsafe_allow_html=True)
    
    # Final Recommendation
    st.markdown("### 📋 Final Recommendation")
    
    if prediction_class == 1:
        st.warning("""
        **🚨 ACTION REQUIRED: DECLINE APPLICATION**
        
        Based on the analysis:
        - High default probability detected
        - Multiple risk factors identified
        - Customer shows concerning payment behavior
        
        **Recommended Actions:**
        1. ❌ Decline credit card application
        2. 💵 Offer secured credit card with deposit
        3. 📞 Recommend credit counseling
        4. 🔄 Re-evaluate after 6-12 months
        """)
    else:
        st.success("""
        **✅ ACTION: APPROVE APPLICATION**
        
        Based on the analysis:
        - Low default probability detected
        - Good payment history
        - Healthy credit utilization
        
        **Recommended Actions:**
        1. ✅ Approve credit card application
        2. 💳 Grant requested credit limit
        3. 📈 Consider limit increase after 6 months
        4. 🎯 Target for premium card offers
        """)

# ==================== HEADER ====================
st.markdown("""
<div class="main-header">
    <h1>💳 Credit Card Default Predictor</h1>
    <p>Advanced Machine Learning Risk Assessment System</p>
</div>
""", unsafe_allow_html=True)

# ==================== MODE SELECTION ====================
mode = st.radio(
    "Select Input Mode",
    ["📝 Single Customer", "📂 Batch CSV Upload"],
    horizontal=True
)

# ==================== SINGLE CUSTOMER MODE ====================
if mode == "📝 Single Customer":
    with st.container():
        st.markdown("### 📋 Customer Information")
        
        tab1, tab2, tab3 = st.tabs(["👤 Personal Info", "📅 Payment History", "💰 Financial Data"])
        
        # Initialize session state for inputs if not exists
        if 'pay_values' not in st.session_state:
            st.session_state.pay_values = [0, 0, 0, 0, 0, 0]
        
        # Tab 1: Personal Information
        with tab1:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                limit_bal = st.number_input(
                    "💵 Credit Limit (NT$)", 
                    min_value=0, 
                    max_value=1000000, 
                    value=200000, 
                    step=10000,
                    key="limit_bal",
                    help="Amount of credit line in New Taiwan Dollars"
                )
                
                age = st.number_input(
                    "🎂 Age", 
                    min_value=18, 
                    max_value=100, 
                    value=35,
                    key="age",
                    help="Customer's age in years"
                )
            
            with col2:
                sex = st.selectbox(
                    "👤 Gender", 
                    ["Female", "Male"],
                    key="sex",
                    help="Customer's gender"
                )
                
                edu_map = {"Graduate School": 1, "University": 2, "High School": 3, "Other": 4}
                education = st.selectbox(
                    "🎓 Education Level", 
                    list(edu_map.keys()),
                    key="education",
                    help="Highest education level attained"
                )
            
            with col3:
                marriage_map = {"Married": 1, "Single": 2, "Other": 3}
                marriage = st.selectbox(
                    "💑 Marital Status", 
                    list(marriage_map.keys()),
                    key="marriage",
                    help="Current marital status"
                )
                
                st.metric("📊 Population Default Rate", "22.1%", help="Overall default rate in training dataset")
        
        # Tab 2: Payment History
        with tab2:
            st.markdown("#### 📅 Repayment Status (Last 6 Months)")
            st.caption("💡 Values: -2=No credit used | -1=Paid in full | 0=Revolving credit | 1-8=Months overdue")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                pay_0 = st.number_input("📆 September (PAY_0)", min_value=-2, max_value=8, value=0, key="pay_0")
                pay_4 = st.number_input("📆 June (PAY_4)", min_value=-2, max_value=8, value=0, key="pay_4")
            
            with col2:
                pay_2 = st.number_input("📆 August (PAY_2)", min_value=-2, max_value=8, value=0, key="pay_2")
                pay_5 = st.number_input("📆 May (PAY_5)", min_value=-2, max_value=8, value=0, key="pay_5")
            
            with col3:
                pay_3 = st.number_input("📆 July (PAY_3)", min_value=-2, max_value=8, value=0, key="pay_3")
                pay_6 = st.number_input("📆 April (PAY_6)", min_value=-2, max_value=8, value=0, key="pay_6")
        
        # Tab 3: Financial Data
        with tab3:
            st.markdown("#### 💰 Bill Statements (NT$)")
            col1, col2 = st.columns(2)
            
            with col1:
                bill_amt1 = st.number_input("📄 September Bill", min_value=0, max_value=1000000, value=50000, step=5000, key="bill_1")
                bill_amt3 = st.number_input("📄 July Bill", min_value=0, max_value=1000000, value=45000, step=5000, key="bill_3")
                bill_amt5 = st.number_input("📄 May Bill", min_value=0, max_value=1000000, value=40000, step=5000, key="bill_5")
            
            with col2:
                bill_amt2 = st.number_input("📄 August Bill", min_value=0, max_value=1000000, value=48000, step=5000, key="bill_2")
                bill_amt4 = st.number_input("📄 June Bill", min_value=0, max_value=1000000, value=43000, step=5000, key="bill_4")
                bill_amt6 = st.number_input("📄 April Bill", min_value=0, max_value=1000000, value=38000, step=5000, key="bill_6")
            
            st.markdown("#### 💸 Previous Payments (NT$)")
            col1, col2 = st.columns(2)
            
            with col1:
                pay_amt1 = st.number_input("💵 September Payment", min_value=0, max_value=500000, value=5000, step=1000, key="pay_amt_1")
                pay_amt3 = st.number_input("💵 July Payment", min_value=0, max_value=500000, value=4500, step=1000, key="pay_amt_3")
                pay_amt5 = st.number_input("💵 May Payment", min_value=0, max_value=500000, value=4000, step=1000, key="pay_amt_5")
            
            with col2:
                pay_amt2 = st.number_input("💵 August Payment", min_value=0, max_value=500000, value=4800, step=1000, key="pay_amt_2")
                pay_amt4 = st.number_input("💵 June Payment", min_value=0, max_value=500000, value=4300, step=1000, key="pay_amt_4")
                pay_amt6 = st.number_input("💵 April Payment", min_value=0, max_value=500000, value=3800, step=1000, key="pay_amt_6")
        
        # ==================== PREDICT BUTTON ====================
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            predict_button = st.button("🎯 PREDICT DEFAULT RISK", use_container_width=True)
        
        # ==================== PREDICTION LOGIC ====================
        if predict_button:
            # Prepare features
            sex_val = 1 if sex == "Male" else 2
            pay_values = [pay_0, pay_2, pay_3, pay_4, pay_5, pay_6]
            bill_amts = [bill_amt1, bill_amt2, bill_amt3, bill_amt4, bill_amt5, bill_amt6]
            pay_amts = [pay_amt1, pay_amt2, pay_amt3, pay_amt4, pay_amt5, pay_amt6]
            
            # Create feature array (23 features)
            features = np.array([[
                limit_bal,
                sex_val,
                edu_map[education],
                marriage_map[marriage],
                age,
                pay_0, pay_2, pay_3, pay_4, pay_5, pay_6,
                bill_amt1, bill_amt2, bill_amt3, bill_amt4, bill_amt5, bill_amt6,
                pay_amt1, pay_amt2, pay_amt3, pay_amt4, pay_amt5, pay_amt6
            ]])
            
            # Get prediction
            probability, prediction_class = predict_single_customer(features, edu_map, marriage_map)
            
            # Calculate risk metrics
            metrics = calculate_risk_metrics(pay_values, bill_amts, pay_amts, limit_bal, age, education, marriage)
            risk_factors = get_risk_factors(metrics, age, education, marriage, limit_bal)
            
            # Display results
            st.divider()
            display_prediction_results(probability, prediction_class, metrics, risk_factors)
            
            # Detailed summary
            with st.expander("📋 View Complete Customer Profile", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**👤 Personal Information**")
                    st.write(f"- Credit Limit: NT$ {limit_bal:,}")
                    st.write(f"- Age: {age}")
                    st.write(f"- Gender: {sex}")
                    st.write(f"- Education: {education}")
                    st.write(f"- Marital Status: {marriage}")
                
                with col2:
                    st.markdown("**📊 Risk Assessment**")
                    st.write(f"- Default Probability: {probability:.1%}")
                    st.write(f"- Prediction: {'DEFAULT' if prediction_class == 1 else 'NO DEFAULT'}")
                    st.write(f"- Risk Score: {metrics['risk_score']}/100")
                    st.write(f"- Late Payments: {metrics['late_payments']}/6")
                    st.write(f"- Credit Utilization: {metrics['utilization']:.1%}")
                    st.write(f"- Payment Ratio: {metrics['pay_ratio']:.1%}")

# ==================== BATCH UPLOAD MODE ====================
else:
    st.markdown("""
    <div class="info-box">
        <h3>📂 Batch Upload Mode</h3>
        <p>Upload a CSV file with customer data to predict default risk for multiple customers at once.</p>
        <p><strong>Required columns:</strong></p>
        <ul>
            <li>LIMIT_BAL, SEX, EDUCATION, MARRIAGE, AGE</li>
            <li>PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6</li>
            <li>BILL_AMT1 through BILL_AMT6</li>
            <li>PAY_AMT1 through PAY_AMT6</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a CSV file", 
        type="csv",
        help="Upload a CSV file with the required columns"
    )
    
    if uploaded_file is not None:
        try:
            # Load and validate data
            df = pd.read_csv(uploaded_file)
            
            # Check required columns
            required_cols = ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
                           'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
                           'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
                           'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
            
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                st.error(f"Missing required columns: {missing_cols}")
                st.stop()
            
            # Map categorical values
            edu_map = {"Graduate School": 1, "University": 2, "High School": 3, "Other": 4}
            marriage_map = {"Married": 1, "Single": 2, "Other": 3}
            
            # If EDUCATION and MARRIAGE are strings, map them
            if df['EDUCATION'].dtype == 'object':
                df['EDUCATION'] = df['EDUCATION'].map(edu_map)
            if df['MARRIAGE'].dtype == 'object':
                df['MARRIAGE'] = df['MARRIAGE'].map(marriage_map)
            
            # Display data preview
            st.markdown("### 📊 Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            st.caption(f"Total: {len(df)} customers")
            
            # Predict button
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                batch_predict = st.button("🎯 RUN BATCH PREDICTION", use_container_width=True)
            
            if batch_predict:
                # Get predictions
                probabilities, predictions = predict_batch(df, edu_map, marriage_map)
                
                # Add predictions to dataframe
                df['Default_Probability'] = probabilities
                df['Prediction'] = predictions
                df['Risk_Level'] = df['Default_Probability'].apply(
                    lambda x: 'High' if x >= 0.65 else ('Medium' if x >= 0.3 else 'Low')
                )
                
                # Summary statistics
                st.markdown("### 📈 Batch Summary Statistics")
                
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.metric("Total Customers", len(df))
                with col2:
                    st.metric("Predicted Defaults", predictions.sum())
                with col3:
                    default_rate = (predictions.sum() / len(df)) * 100
                    st.metric("Default Rate", f"{default_rate:.1f}%")
                with col4:
                    avg_prob = probabilities.mean() * 100
                    st.metric("Avg Risk Probability", f"{avg_prob:.1f}%")
                with col5:
                    high_risk = (df['Default_Probability'] >= 0.65).sum()
                    st.metric("High Risk Customers", high_risk)
                
                # Risk distribution charts
                st.markdown("### 📊 Risk Distribution")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Pie chart for predictions
                    pred_counts = df['Prediction'].value_counts()
                    fig_pie = go.Figure(data=[go.Pie(
                        labels=['No Default', 'Default'],
                        values=[pred_counts.get(0, 0), pred_counts.get(1, 0)],
                        marker=dict(colors=['#10b981', '#dc2626']),
                        hole=0.4
                    )])
                    fig_pie.update_layout(title="Prediction Distribution", height=400)
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with col2:
                    # Histogram of probabilities
                    fig_hist = go.Figure(data=[go.Histogram(
                        x=df['Default_Probability'],
                        nbinsx=30,
                        marker=dict(
                            color=df['Default_Probability'],
                            colorscale='RdYlGn_r',
                            showscale=True
                        )
                    )])
                    fig_hist.update_layout(
                        title="Default Probability Distribution",
                        xaxis_title="Default Probability",
                        yaxis_title="Number of Customers",
                        height=400
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)
                
                # Risk level breakdown
                risk_counts = df['Risk_Level'].value_counts()
                risk_colors = {'Low': '#10b981', 'Medium': '#f59e0b', 'High': '#dc2626'}
                
                fig_bar = go.Figure(data=[go.Bar(
                    x=risk_counts.index,
                    y=risk_counts.values,
                    marker_color=[risk_colors[level] for level in risk_counts.index]
                )])
                fig_bar.update_layout(
                    title="Risk Level Distribution",
                    xaxis_title="Risk Level",
                    yaxis_title="Number of Customers",
                    height=400
                )
                st.plotly_chart(fig_bar, use_container_width=True)
                
                # Detailed results table
                st.markdown("### 📋 Detailed Predictions")
                
                # Format display columns
                display_cols = ['LIMIT_BAL', 'AGE', 'Default_Probability', 'Prediction', 'Risk_Level']
                available_cols = [col for col in display_cols if col in df.columns]
                
                # Add additional informative columns if available
                if 'EDUCATION' in df.columns:
                    if df['EDUCATION'].dtype == 'object':
                        available_cols.insert(2, 'EDUCATION')
                if 'MARRIAGE' in df.columns:
                    if df['MARRIAGE'].dtype == 'object':
                        available_cols.insert(3, 'MARRIAGE')
                
                display_df = df[available_cols].copy()
                display_df['Default_Probability'] = display_df['Default_Probability'].apply(lambda x: f"{x:.1%}")
                display_df['Prediction'] = display_df['Prediction'].map({0: 'No Default', 1: 'Default'})
                
                st.dataframe(display_df, use_container_width=True)
                
                # Download results
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Predictions (CSV)",
                    data=csv,
                    file_name=f"credit_default_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                )
                
                # High risk customers alert
                high_risk_df = df[df['Default_Probability'] >= 0.65]
                if len(high_risk_df) > 0:
                    st.warning(f"⚠️ {len(high_risk_df)} customers identified as HIGH RISK. Review these customers carefully.")
                    
                    with st.expander("🔴 View High Risk Customers"):
                        high_risk_display = high_risk_df[['LIMIT_BAL', 'AGE', 'Default_Probability']].copy()
                        high_risk_display['Default_Probability'] = high_risk_display['Default_Probability'].apply(lambda x: f"{x:.1%}")
                        st.dataframe(high_risk_display, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error processing file: {e}")
            st.info("Please ensure your CSV file has the correct format and column names.")

# ==================== FOOTER ====================
st.markdown("""
<div class="footer">
    <p>Credit Card Default Predictor | Powered by Logistic Regression | Trained on UCI Credit Card Dataset</p>
    <p>© 2024 - Professional Risk Assessment System</p>
</div>
""", unsafe_allow_html=True)