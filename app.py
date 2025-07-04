import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go
from io import StringIO
import chardet
from sklearn.metrics import mean_squared_error, r2_score

# Page config - Sets up the layout and icon
st.set_page_config(
    page_title="Quantum-AI Supply Chain Optimizer",
    page_icon="🔮",
    layout="wide"
)

# Custom CSS for a professional look
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Header with styled title
st.markdown('<div class="main-header">🔮 Quantum-AI Supply Chain Optimizer</div>', unsafe_allow_html=True)

# Function to get icon based on feature name
def get_icon(feature):
    """Return an icon based on feature name."""
    icons = {
        'footfall': '👥',
        'expiry': '📅',
        'price': '💰',
        'weather': '🌡️'
    }
    return icons.get(feature.lower(), '📊')

# Sidebar for data source selection and inputs
st.sidebar.header("🎛️ Data and Input Parameters")
st.sidebar.markdown("Choose your data source and adjust parameters for predictions:")

# Data source selection
data_source = st.sidebar.radio("Select Data Source:", ("Default (sample_data.csv)", "Upload New CSV"))
uploaded_file = None
df = None

def detect_encoding(file_path_or_buffer):
    """Detect the encoding of the file or buffer."""
    try:
        raw_data = file_path_or_buffer.read() if hasattr(file_path_or_buffer, 'read') else open(file_path_or_buffer, 'rb').read()
        result = chardet.detect(raw_data)
        if hasattr(file_path_or_buffer, 'seek'): file_path_or_buffer.seek(0)
        return result['encoding']
    except Exception as e:
        st.error(f"❌ Error detecting encoding: {e}")
        return None

def generate_target_column(df):
    """Generate a target column dynamically if missing."""
    if 'target' not in df.columns:
        st.warning("⚠️ 'target' column not found. Generating dynamically...")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            base_col = numeric_cols[0]
            df['target'] = df[base_col] * (1 + np.random.uniform(-0.1, 0.1))
            st.success(f"✅ Target created based on {base_col} with random variation.")
        else:
            st.error("❌ No numeric columns available to generate target. Please upload data with numeric fields.")
            st.stop()
    return df

# Load data based on source
if data_source == "Upload New CSV":
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file:
        encoding = detect_encoding(uploaded_file)
        if encoding:
            df = pd.read_csv(uploaded_file, encoding=encoding)
            df = generate_target_column(df)
            st.sidebar.success(f"✅ File uploaded (Encoding: {encoding})")
        else:
            st.sidebar.error("❌ Encoding detection failed.")
    else:
        st.sidebar.warning("⚠️ Upload a CSV to proceed.")
else:
    try:
        df = pd.read_csv("data/sample_data.csv", encoding=detect_encoding("data/sample_data.csv"))
        df = generate_target_column(df)
        st.sidebar.success("✅ Using default data: sample_data.csv")
    except (FileNotFoundError, UnicodeDecodeError) as e:
        st.sidebar.error(f"❌ Default file error: {e}. Upload a CSV or fix sample_data.csv.")
        st.stop()


try:
    model, features = joblib.load("models/xgb_model.pkl")
    st.success("✅ Model loaded!")
except FileNotFoundError:
    st.error("❌ Model not found. Run 'python run_pipeline.py' to train.")
    st.code("python run_pipeline.py")
    st.stop()

user_input = {}
if df is not None:
    for feature in features:
        if feature in df.columns and feature in df.select_dtypes(include=[np.number]).columns:
            user_input[feature] = st.sidebar.slider(
                f"{get_icon(feature)} {feature.title()}",
                int(df[feature].min()), int(df[feature].max()), int(df[feature].mean())
            )
        else:
            user_input[feature] = st.sidebar.slider(
                f"📊 {feature.title()} (Other Factors)", 0.0, 100.0, 50.0
            )


col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📈 Demand Prediction")
    if df is not None and user_input:
        input_df = pd.DataFrame([user_input])
        available_features = [f for f in features if f in df.select_dtypes(include=[np.number]).columns]
        if available_features:
            input_df = input_df[available_features].reindex(columns=features, fill_value=0)
            prediction = model.predict(input_df)[0]

            X = df[features].fillna(0)
            y = df['target']
            y_pred = model.predict(X)
            mse = mean_squared_error(y, y_pred)
            r2 = r2_score(y, y_pred)
        else:
            st.error("❌ No numeric features for prediction.")
            st.stop()


        st.markdown(f"""
        <div class="metric-card">
            <h2>Predicted Demand</h2>
            <h1>{prediction:.1f} units</h1>
        </div>
        """, unsafe_allow_html=True)


        st.markdown(f"""
        <div class="metric-card">
            <h2>Model Performance (R²)</h2>
            <h1>{r2:.2f}</h1>
        </div>
        """, unsafe_allow_html=True)


        st.subheader("🔍 Feature Importance")
        fig = go.Figure(data=[go.Bar(x=features, y=model.feature_importances_, marker_color='rgb(102, 126, 234)')])
        fig.update_layout(title="Feature Importance", xaxis_title="Features", yaxis_title="Importance Score", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("📊 Input Summary")
    for feature, value in user_input.items():
        st.metric(label=feature.replace('_', ' ').title(), value=f"{value:.1f}", delta=None)

    st.subheader("🧠 Quick Analysis")
    if prediction > 250:
        st.success("🔥 High demand expected!")
        st.info("💡 Consider increasing inventory")
    elif prediction > 200:
        st.warning("📊 Moderate demand")
        st.info("💡 Standard inventory levels")
    else:
        st.error("📉 Low demand predicted")
        st.info("💡 Consider promotions")


st.markdown("---")
st.markdown("🚀 **Powered by Quantum-Inspired AI Optimization**")
with st.expander("ℹ️ Need Help? Click Here"):
    st.write("""
    **Welcome!** This tool predicts demand. Use 'Default' or upload a CSV. Adjust sliders for features. 
    R² shows model fit (1.0 is perfect). Run 'python run_pipeline.py' if the model is missing. Ask for help if needed!
    """)