import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
import json
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Page configuration
st.set_page_config(
    page_title="PayPal Fraud Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .fraud-alert {
        background-color: #ffebee;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #f44336;
        color: #d32f2f;
    }
    .legitimate-alert {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #4caf50;
        color: #388e3c;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    .prediction-box {
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .fraud-prediction {
        background: linear-gradient(45deg, #ff6b6b, #ffa07a);
        color: white;
        border: 2px solid #ff4757;
    }
    .legitimate-prediction {
        background: linear-gradient(45deg, #51cf66, #69db7c);
        color: white;
        border: 2px solid #2ed573;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained model and preprocessing components."""
    try:
        with open('paypal_fraud_detection_model.pkl', 'rb') as file:
            model_package = pickle.load(file)
        return model_package
    except FileNotFoundError:
        st.error("❌ Model file not found. Please ensure 'paypal_fraud_detection_model.pkl' is in the same directory.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()

def create_sample_data():
    """Create sample transaction data for demonstration."""
    np.random.seed(42)
    sample_data = []
    
    transaction_types = ['Payment', 'Transfer', 'Withdrawal', 'Deposit', 'Bill Payment', 'Refund']
    countries = ['USA', 'Canada', 'UK', 'France', 'Germany', 'Australia', 'India', 'China', 'Brazil', 'Nigeria']
    device_types = ['Desktop', 'Mobile', 'Tablet']
    times_of_day = ['Morning', 'Afternoon', 'Evening', 'Night']
    payment_gateways = ['PayPal', 'Stripe', 'Square', 'Razorpay']
    
    for i in range(20):
        sample_data.append({
            'transaction_id': f'TXN{10001+i:05d}',
            'user_id': f'U{np.random.randint(1000, 9999)}',
            'amount': round(np.random.lognormal(4, 1.5), 2),
            'transaction_type': np.random.choice(transaction_types),
            'account_age_days': np.random.randint(30, 3650),
            'country': np.random.choice(countries),
            'device_type': np.random.choice(device_types),
            'ip_address_risk': np.random.randint(1, 100),
            'time_of_day': np.random.choice(times_of_day),
            'num_prev_transactions': np.random.randint(1, 1000),
            'avg_transaction_value': round(np.random.lognormal(5, 1), 2),
            'is_foreign_transaction': np.random.randint(0, 2),
            'is_high_risk_country': np.random.randint(0, 2),
            'is_vpn_used': np.random.randint(0, 2),
            'login_attempts': np.random.randint(1, 10),
            'device_trust_score': np.random.randint(1, 100),
            'payment_gateway': np.random.choice(payment_gateways),
            'browser_fingerprint_score': np.random.randint(1, 100),
            'session_duration_sec': np.random.randint(10, 2000)
        })
    
    return pd.DataFrame(sample_data)

def preprocess_input(data, model_package):
    """Preprocess input data for prediction."""
    df = data.copy()
    
    # Get label encoders from model package
    label_encoders = model_package['label_encoders']
    
    # Encode categorical variables
    categorical_features = ['transaction_type', 'country', 'device_type', 'time_of_day', 'payment_gateway']
    
    for feature in categorical_features:
        if feature in label_encoders:
            le = label_encoders[feature]
            # Handle unseen categories
            df[feature + '_encoded'] = df[feature].map(
                lambda x: le.transform([x])[0] if x in le.classes_ else 0
            )
        else:
            df[feature + '_encoded'] = 0
    
    # Create engineered features
    df['amount_bin'] = pd.cut(df['amount'], 
                             bins=[0, 50, 200, 500, float('inf')], 
                             labels=[0, 1, 2, 3])
    df['amount_bin_encoded'] = df['amount_bin'].astype(int)
    
    df['account_age_category'] = pd.cut(df['account_age_days'], 
                                       bins=[0, 30, 365, 1095, float('inf')], 
                                       labels=[0, 1, 2, 3])
    df['account_age_category_encoded'] = df['account_age_category'].astype(int)
    
    # Composite risk score
    df['composite_risk_score'] = (
        df['ip_address_risk'] * 0.3 +
        (100 - df['device_trust_score']) * 0.25 +
        df['browser_fingerprint_score'] * 0.2 +
        df['login_attempts'] * 10 * 0.15 +
        df['is_vpn_used'] * 50 * 0.1
    )
    
    # Transaction velocity
    df['transaction_velocity'] = df['num_prev_transactions'] / (df['account_age_days'] + 1)
    
    # Select required features
    feature_columns = model_package['feature_columns']
    return df[feature_columns]

def predict_fraud(data, model_package):
    """Make fraud predictions using the loaded model."""
    model = model_package['model']
    scaler = model_package['scaler']
    model_name = model_package['model_name']
    
    # Preprocess the data
    X = preprocess_input(data, model_package)
    
    # Apply scaling if needed (for Logistic Regression)
    if model_name == 'Logistic Regression' and scaler is not None:
        X_processed = scaler.transform(X)
    else:
        X_processed = X
    
    # Make predictions
    predictions = model.predict(X_processed)
    probabilities = model.predict_proba(X_processed)
    
    return predictions, probabilities

def create_risk_gauge(risk_score):
    """Create a risk gauge visualization."""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = risk_score * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Fraud Risk Score"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 80
            }
        }
    ))
    
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">🛡️ PayPal Fraud Detection System</h1>', unsafe_allow_html=True)
    
    # Load model
    model_package = load_model()
    
    # Sidebar
    st.sidebar.title("🔧 Control Panel")
    
    # Mode selection
    mode = st.sidebar.selectbox(
        "📊 Select Mode",
        ["Single Transaction Analysis", "Batch Analysis", "Model Information", "Demo Data"]
    )
    
    if mode == "Single Transaction Analysis":
        st.header("🔍 Single Transaction Analysis")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("💳 Transaction Details")
            
            # Create input form
            with st.form("transaction_form"):
                # Basic transaction info
                col_a, col_b = st.columns(2)
                
                with col_a:
                    transaction_id = st.text_input("Transaction ID", value="TXN12345")
                    user_id = st.text_input("User ID", value="U1234")
                    amount = st.number_input("Amount ($)", min_value=0.01, value=150.00, step=0.01)
                    transaction_type = st.selectbox("Transaction Type", 
                                                   ['Payment', 'Transfer', 'Withdrawal', 'Deposit', 'Bill Payment', 'Refund'])
                    account_age_days = st.number_input("Account Age (days)", min_value=1, value=365)
                
                with col_b:
                    country = st.selectbox("Country", 
                                         ['USA', 'Canada', 'UK', 'France', 'Germany', 'Australia', 'India', 'China', 'Brazil', 'Nigeria'])
                    device_type = st.selectbox("Device Type", ['Desktop', 'Mobile', 'Tablet'])
                    time_of_day = st.selectbox("Time of Day", ['Morning', 'Afternoon', 'Evening', 'Night'])
                    payment_gateway = st.selectbox("Payment Gateway", ['PayPal', 'Stripe', 'Square', 'Razorpay'])
                    num_prev_transactions = st.number_input("Previous Transactions", min_value=0, value=50)
                
                # Risk factors
                st.subheader("⚠️ Risk Factors")
                col_c, col_d = st.columns(2)
                
                with col_c:
                    ip_address_risk = st.slider("IP Address Risk (1-100)", 1, 100, 25)
                    device_trust_score = st.slider("Device Trust Score (1-100)", 1, 100, 75)
                    browser_fingerprint_score = st.slider("Browser Fingerprint Score (1-100)", 1, 100, 30)
                    login_attempts = st.number_input("Login Attempts", min_value=1, max_value=10, value=2)
                
                with col_d:
                    avg_transaction_value = st.number_input("Avg Transaction Value ($)", min_value=0.01, value=200.00)
                    session_duration_sec = st.number_input("Session Duration (seconds)", min_value=1, value=300)
                    is_foreign_transaction = st.checkbox("Foreign Transaction")
                    is_high_risk_country = st.checkbox("High Risk Country")
                    is_vpn_used = st.checkbox("VPN Used")
                
                submit_button = st.form_submit_button("🔍 Analyze Transaction", use_container_width=True)
        
        if submit_button:
            # Create transaction data
            transaction_data = pd.DataFrame([{
                'transaction_id': transaction_id,
                'user_id': user_id,
                'amount': amount,
                'transaction_type': transaction_type,
                'account_age_days': account_age_days,
                'country': country,
                'device_type': device_type,
                'ip_address_risk': ip_address_risk,
                'time_of_day': time_of_day,
                'num_prev_transactions': num_prev_transactions,
                'avg_transaction_value': avg_transaction_value,
                'is_foreign_transaction': int(is_foreign_transaction),
                'is_high_risk_country': int(is_high_risk_country),
                'is_vpn_used': int(is_vpn_used),
                'login_attempts': login_attempts,
                'device_trust_score': device_trust_score,
                'payment_gateway': payment_gateway,
                'browser_fingerprint_score': browser_fingerprint_score,
                'session_duration_sec': session_duration_sec
            }])
            
            # Make prediction
            predictions, probabilities = predict_fraud(transaction_data, model_package)
            
            fraud_probability = probabilities[0][1]
            is_fraud = predictions[0]
            
            with col2:
                st.subheader("📊 Analysis Results")
                
                # Prediction result box
                if is_fraud:
                    st.markdown(f'''
                    <div class="prediction-box fraud-prediction">
                        🚨 FRAUD DETECTED<br>
                        Risk Level: HIGH<br>
                        Confidence: {fraud_probability:.1%}
                    </div>
                    ''', unsafe_allow_html=True)
                else:
                    st.markdown(f'''
                    <div class="prediction-box legitimate-prediction">
                        ✅ LEGITIMATE<br>
                        Risk Level: LOW<br>
                        Confidence: {(1-fraud_probability):.1%}
                    </div>
                    ''', unsafe_allow_html=True)
                
                # Risk gauge
                st.plotly_chart(create_risk_gauge(fraud_probability), use_container_width=True)
                
                # Risk breakdown
                st.subheader("🎯 Risk Factors")
                risk_factors = {
                    "IP Risk": ip_address_risk / 100,
                    "Device Trust": (100 - device_trust_score) / 100,
                    "Browser Score": browser_fingerprint_score / 100,
                    "Login Attempts": min(login_attempts / 10, 1),
                    "VPN Usage": int(is_vpn_used)
                }
                
                for factor, score in risk_factors.items():
                    st.metric(factor, f"{score:.1%}", 
                             delta=f"{'High' if score > 0.5 else 'Low'} Risk")
    
    elif mode == "Batch Analysis":
        st.header("📊 Batch Transaction Analysis")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload CSV file with transaction data", 
            type=['csv'],
            help="Upload a CSV file containing transaction data with the required columns."
        )
        
        if uploaded_file is not None:
            try:
                # Read uploaded data
                batch_data = pd.read_csv(uploaded_file)
                st.success(f"✅ Loaded {len(batch_data)} transactions")
                
                # Show data preview
                st.subheader("📋 Data Preview")
                st.dataframe(batch_data.head(), use_container_width=True)
                
                if st.button("🔍 Analyze Batch", use_container_width=True):
                    with st.spinner("Analyzing transactions..."):
                        # Make predictions
                        predictions, probabilities = predict_fraud(batch_data, model_package)
                        
                        # Add results to dataframe
                        results_df = batch_data.copy()
                        results_df['fraud_prediction'] = predictions
                        results_df['fraud_probability'] = probabilities[:, 1]
                        results_df['risk_level'] = pd.cut(
                            results_df['fraud_probability'], 
                            bins=[0, 0.3, 0.7, 1.0], 
                            labels=['Low', 'Medium', 'High']
                        )
                        
                        # Summary statistics
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Total Transactions", len(results_df))
                        with col2:
                            fraud_count = sum(predictions)
                            st.metric("Flagged as Fraud", fraud_count, delta=f"{fraud_count/len(results_df):.1%}")
                        with col3:
                            avg_risk = results_df['fraud_probability'].mean()
                            st.metric("Average Risk Score", f"{avg_risk:.1%}")
                        with col4:
                            high_risk = sum(results_df['risk_level'] == 'High')
                            st.metric("High Risk Transactions", high_risk)
                        
                        # Visualizations
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Risk distribution
                            fig1 = px.histogram(
                                results_df, 
                                x='fraud_probability', 
                                nbins=20,
                                title="Fraud Risk Distribution",
                                labels={'fraud_probability': 'Fraud Probability', 'count': 'Number of Transactions'}
                            )
                            st.plotly_chart(fig1, use_container_width=True)
                        
                        with col2:
                            # Risk level pie chart
                            risk_counts = results_df['risk_level'].value_counts()
                            fig2 = px.pie(
                                values=risk_counts.values, 
                                names=risk_counts.index,
                                title="Risk Level Distribution",
                                color_discrete_map={'Low': 'green', 'Medium': 'yellow', 'High': 'red'}
                            )
                            st.plotly_chart(fig2, use_container_width=True)
                        
                        # Detailed results
                        st.subheader("📊 Detailed Results")
                        
                        # Filter options
                        risk_filter = st.selectbox("Filter by Risk Level", ['All', 'High', 'Medium', 'Low'])
                        
                        if risk_filter != 'All':
                            display_df = results_df[results_df['risk_level'] == risk_filter]
                        else:
                            display_df = results_df
                        
                        # Display results
                        display_columns = ['transaction_id', 'amount', 'country', 'transaction_type', 
                                         'fraud_prediction', 'fraud_probability', 'risk_level']
                        st.dataframe(
                            display_df[display_columns].style.format({
                                'fraud_probability': '{:.1%}',
                                'amount': '${:.2f}'
                            }),
                            use_container_width=True
                        )
                        
                        # Download results
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results as CSV",
                            data=csv,
                            file_name=f"fraud_analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime='text/csv',
                            use_container_width=True
                        )
                        
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
        
        else:
            # Show sample data option
            st.info("👆 Upload a CSV file to analyze multiple transactions, or use the demo data below.")
            
            if st.button("🎲 Use Demo Data", use_container_width=True):
                sample_data = create_sample_data()
                st.session_state['demo_data'] = sample_data
                st.rerun()
            
            if 'demo_data' in st.session_state:
                st.subheader("📊 Demo Data Analysis")
                demo_data = st.session_state['demo_data']
                
                st.dataframe(demo_data.head(10), use_container_width=True)
                
                if st.button("🔍 Analyze Demo Data", use_container_width=True):
                    with st.spinner("Analyzing demo transactions..."):
                        predictions, probabilities = predict_fraud(demo_data, model_package)
                        
                        # Create results visualization
                        results_df = demo_data.copy()
                        results_df['fraud_prediction'] = predictions
                        results_df['fraud_probability'] = probabilities[:, 1]
                        
                        # Summary
                        fraud_count = sum(predictions)
                        avg_risk = results_df['fraud_probability'].mean()
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Transactions", len(results_df))
                        with col2:
                            st.metric("Flagged as Fraud", fraud_count, delta=f"{fraud_count/len(results_df):.1%}")
                        with col3:
                            st.metric("Average Risk Score", f"{avg_risk:.1%}")
                        
                        # Results table
                        st.subheader("📊 Analysis Results")
                        display_df = results_df[['transaction_id', 'amount', 'country', 'transaction_type', 
                                               'fraud_prediction', 'fraud_probability']].copy()
                        display_df['fraud_prediction'] = display_df['fraud_prediction'].map({0: '✅ Legitimate', 1: '🚨 Fraud'})
                        
                        st.dataframe(
                            display_df.style.format({
                                'fraud_probability': '{:.1%}',
                                'amount': '${:.2f}'
                            }),
                            use_container_width=True
                        )
    
    elif mode == "Model Information":
        st.header("🤖 Model Information")
        
        # Model details
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Model Performance")
            metrics = model_package['performance_metrics']
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Accuracy", f"{metrics['accuracy']:.1%}")
                st.metric("Precision", f"{metrics['precision']:.1%}")
            with col_b:
                st.metric("Recall", f"{metrics['recall']:.1%}")
                st.metric("F1-Score", f"{metrics['f1_score']:.1%}")
            
            st.metric("AUC-ROC", f"{metrics['auc_roc']:.3f}")
        
        with col2:
            st.subheader("🔧 Model Details")
            st.info(f"**Model Type:** {model_package['model_name']}")
            st.info(f"**Training Date:** {model_package['training_date']}")
            st.info(f"**Dataset Size:** {model_package['dataset_shape'][0]:,} transactions")
            st.info(f"**Features:** {len(model_package['feature_columns'])} features")
        
        # Feature importance
        if model_package['feature_importance']:
            st.subheader("🎯 Feature Importance")
            
            importance_df = pd.DataFrame(
                list(model_package['feature_importance'].items()),
                columns=['Feature', 'Importance']
            ).sort_values('Importance', ascending=True).tail(15)
            
            fig = px.bar(
                importance_df, 
                x='Importance', 
                y='Feature',
                orientation='h',
                title="Top 15 Most Important Features",
                labels={'Importance': 'Feature Importance Score'}
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
        
        # Feature list
        st.subheader("📋 Model Features")
        features_df = pd.DataFrame({
            'Feature': model_package['feature_columns'],
            'Type': ['Numerical' if 'encoded' not in f else 'Categorical' for f in model_package['feature_columns']]
        })
        st.dataframe(features_df, use_container_width=True)
    
    elif mode == "Demo Data":
        st.header("🎲 Demo Data Generator")
        
        st.write("Generate sample transaction data for testing the fraud detection system.")
        
        num_samples = st.slider("Number of samples to generate", 5, 100, 20)
        
        if st.button("🎲 Generate Demo Data", use_container_width=True):
            demo_data = create_sample_data()
            demo_data = demo_data.head(num_samples)
            
            st.subheader("📊 Generated Demo Data")
            st.dataframe(demo_data, use_container_width=True)
            
            # Quick analysis
            if st.button("🔍 Quick Analysis", use_container_width=True):
                predictions, probabilities = predict_fraud(demo_data, model_package)
                
                # Add predictions to display
                display_df = demo_data.copy()
                display_df['Fraud Risk'] = probabilities[:, 1]
                display_df['Prediction'] = ['🚨 Fraud' if p == 1 else '✅ Legitimate' for p in predictions]
                
                st.subheader("📊 Analysis Results")
                st.dataframe(
                    display_df[['transaction_id', 'amount', 'country', 'Prediction', 'Fraud Risk']].style.format({
                        'Fraud Risk': '{:.1%}',
                        'amount': '${:.2f}'
                    }),
                    use_container_width=True
                )
                
                # Summary stats
                fraud_count = sum(predictions)
                avg_risk = probabilities[:, 1].mean()
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Samples", len(demo_data))
                with col2:
                    st.metric("Flagged as Fraud", fraud_count, delta=f"{fraud_count/len(demo_data):.1%}")
                with col3:
                    st.metric("Average Risk", f"{avg_risk:.1%}")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; padding: 1rem;'>
            🛡️ PayPal Fraud Detection System | Built with Streamlit | 
            Powered by Machine Learning
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()