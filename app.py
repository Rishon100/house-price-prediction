import streamlit as st
import pandas as pd
import joblib

# Load model and features
model = joblib.load("model.pkl")
features = joblib.load("features.pkl")

# Page config
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        font-weight: bold;
        padding: 0.75rem 1rem;
        border-radius: 10px;
        border: none;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .section-header {
        font-size: 1.2rem;
        font-weight: 600;
        color: #1f77b4;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
        padding-bottom: 0.3rem;
        border-bottom: 2px solid #e0e0e0;
    }
    div[data-testid="stNumberInput"] input {
        border-radius: 8px;
    }
    div[data-testid="stSelectbox"] select {
        border-radius: 8px;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    }
    .prediction-amount {
        font-size: 2.5rem;
        font-weight: bold;
        color: white;
        margin: 1rem 0;
    }
    .prediction-label {
        font-size: 1.2rem;
        color: #f0f0f0;
        margin-bottom: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Header
st.title("🏠 House Price Prediction")
st.markdown(
    "Get instant house price estimates using a trained Machine Learning model. "
    "Simply enter the property details below."
)

st.divider()

# Property Details Section
st.markdown('<div class="section-header">📏 Property Specifications</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    area = st.number_input("Area (sqft)", min_value=0, value=None, step=100, help="Total area in square feet", placeholder="e.g. 3000")

with col2:
    bedrooms = st.number_input("Bedrooms", min_value=0, max_value=10, value=None, step=1, placeholder="e.g. 3")

with col3:
    bathrooms = st.number_input("Bathrooms", min_value=0, max_value=10, value=None, step=1, placeholder="e.g. 2")

col4, col5 = st.columns(2)

with col4:
    stories = st.number_input("Stories", min_value=1, max_value=10, value=None, step=1, help="Number of floors", placeholder="e.g. 2")

with col5:
    parking = st.number_input("Parking Spaces", min_value=0, max_value=10, value=None, step=1, placeholder="e.g. 1")

st.markdown("")  # Spacing

# Amenities Section
st.markdown('<div class="section-header">✨ Amenities & Features</div>', unsafe_allow_html=True)

col6, col7, col8 = st.columns(3)

with col6:
    mainroad = st.selectbox("Main Road", ["yes", "no"], index=0, help="Connected to main road")
    guestroom = st.selectbox("Guest Room", ["yes", "no"], index=1)

with col7:
    basement = st.selectbox("Basement", ["yes", "no"], index=1)
    hotwaterheating = st.selectbox("Hot Water", ["yes", "no"], index=1)

with col8:
    airconditioning = st.selectbox("AC", ["yes", "no"], index=0)
    prefarea = st.selectbox("Preferred Area", ["yes", "no"], index=0, help="Located in preferred neighborhood")

st.markdown("")  # Spacing

# Furnishing Section
st.markdown('<div class="section-header">🛋️ Furnishing Details</div>', unsafe_allow_html=True)

furnishingstatus = st.radio(
    "Furnishing Status",
    ["furnished", "semi-furnished", "unfurnished"],
    horizontal=True,
    index=0
)

st.divider()

# Predict Button
if st.button("🔮 Predict House Price", use_container_width=True):
    # Validate inputs
    if area is None or area <= 0:
        st.error("⚠️ Please enter a valid area.")
    elif bedrooms is None:
        st.error("⚠️ Please enter number of bedrooms.")
    elif bathrooms is None:
        st.error("⚠️ Please enter number of bathrooms.")
    elif stories is None:
        st.error("⚠️ Please enter number of stories.")
    elif parking is None:
        st.error("⚠️ Please enter number of parking spaces.")
    else:
        with st.spinner("Analyzing property details..."):
            input_data = {
                "area": int(area),
                "bedrooms": int(bedrooms),
                "bathrooms": int(bathrooms),
                "stories": int(stories),
                "parking": int(parking),
                "mainroad": mainroad,
                "guestroom": guestroom,
                "basement": basement,
                "hotwaterheating": hotwaterheating,
                "airconditioning": airconditioning,
                "prefarea": prefarea,
                "furnishingstatus": furnishingstatus
            }

            # Create DataFrame and prepare for prediction
            df = pd.DataFrame([input_data])
            df = pd.get_dummies(df)
            df = df.reindex(columns=features, fill_value=0)

            # Make prediction
            prediction = model.predict(df)[0]

            # Display result with custom styling
            st.markdown(f"""
                <div class="prediction-box">
                    <div class="prediction-label">💰 Estimated House Price</div>
                    <div class="prediction-amount">₹{int(prediction):,}</div>
                    <p style="color: #f0f0f0; font-size: 0.9rem; margin-top: 1rem;">
                        This prediction is based on historical housing data and trained ML models
                    </p>
                </div>
                """, unsafe_allow_html=True)

            # Additional insights
            st.info(f"""
            **Property Summary:**
            - **Size:** {int(area):,} sqft with {int(bedrooms)} bedrooms and {int(bathrooms)} bathrooms
            - **Type:** {int(stories)}-story {furnishingstatus} home
            - **Price per sqft:** ₹{int(prediction/area):,}
            """)

# Footer
st.markdown("---")
st.caption("🤖 Powered by Machine Learning | Built with Streamlit")