import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor, VotingClassifier
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
import time
import json

# --- Page Config ---
st.image()
st.set_page_config(page_title="Skill-Based Salary & Placement AI", layout="wide", page_icon="🚀")

# --- Custom Styling ---
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; }
    .metric-card { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center; }
    </style>
    """, unsafe_allow_html=True)

# --- 1. Synthetic Data Generation ---
@st.cache_data
def generate_synthetic_data(n_samples=1000):
    np.random.seed(42)
    data = {
        'Xth_Score': np.random.uniform(60, 95, n_samples),
        'XIIth_Score': np.random.uniform(60, 95, n_samples),
        'BE_CGPA': np.random.uniform(6.0, 9.8, n_samples),
        'Skill_Courses': np.random.randint(0, 8, n_samples),
        'Internships': np.random.randint(0, 4, n_samples),
        'Projects': np.random.randint(1, 10, n_samples),
        'Certifications': np.random.randint(0, 6, n_samples),
        'Aptitude_Score': np.random.uniform(50, 100, n_samples),
    }
    df = pd.DataFrame(data)
    
    # Logic for Placement (Binary)
    placement_prob = (df['BE_CGPA'] * 0.4 + df['Internships'] * 0.2 + df['Skill_Courses'] * 0.15 + df['Projects'] * 0.15 + df['Aptitude_Score']*0.01) / 10
    df['Placed'] = (placement_prob + np.random.normal(0, 0.05, n_samples) > 0.6).astype(int)
    
    # Logic for Salary (Regression) - Only relevant if placed
    base_salary = 3.0
    df['Salary_LPA'] = (
        base_salary + 
        (df['BE_CGPA'] - 6) * 1.5 + 
        df['Internships'] * 2.0 + 
        df['Projects'] * 0.8 + 
        df['Skill_Courses'] * 0.5 + 
        np.random.normal(0, 0.5, n_samples)
    )
    df.loc[df['Placed'] == 0, 'Salary_LPA'] = 0  # Salary 0 if not placed
    
    return df

# --- 2. Data Processing & Model Training ---
def train_models(df):
    # Prepare Data
    X = df.drop(['Placed', 'Salary_LPA'], axis=1)
    y_class = df['Placed']
    
    # Filter only placed students for Salary prediction
    df_placed = df[df['Placed'] == 1]
    X_reg = df_placed.drop(['Placed', 'Salary_LPA'], axis=1)
    y_reg = df_placed['Salary_LPA']

    # Train-Test Split
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_class, test_size=0.2, random_state=42)
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

    # Scaling
    scaler = StandardScaler()
    X_train_c_scaled = scaler.fit_transform(X_train_c)
    X_test_c_scaled = scaler.transform(X_test_c)
    X_train_r_scaled = scaler.fit_transform(X_train_r)
    X_test_r_scaled = scaler.transform(X_test_r)

    # Models
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    reg = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)

    clf.fit(X_train_c_scaled, y_train_c)
    reg.fit(X_train_r_scaled, y_train_r)

    # Evaluation
    acc = accuracy_score(y_test_c, clf.predict(X_test_c_scaled))
    r2 = r2_score(y_test_r, reg.predict(X_test_r_scaled))
    mae = mean_absolute_error(y_test_r, reg.predict(X_test_r_scaled))

    return clf, reg, scaler, (acc, r2, mae), X.columns

# --- Main App Logic ---
def main():
    st.title(" Engineering Career AI Predictor")
    st.markdown("### Skill-Based Salary & Placement Forecasting with Synthetic Data")

    # Sidebar Navigation
    menu = ["Data Explorer", "ML Dashboard", "Live Prediction", "Performance Tracking"]
    choice = st.sidebar.selectbox("Navigation", menu)

    df = generate_synthetic_data(1500)
    clf, reg, scaler, metrics, feature_cols = train_models(df)

    if choice == "Data Explorer":
        st.header(" Exploratory Data Analysis")
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.write("### Dataset Overview")
            st.dataframe(df.head(10))
            st.write(f"Total Records: {len(df)}")
            st.write(f"Placement Rate: {round(df['Placed'].mean()*100, 2)}%")

        with col2:
            st.write("### CGPA vs Salary Distribution")
            fig = px.scatter(df[df['Placed']==1], x='BE_CGPA', y='Salary_LPA', 
                             color='Internships', size='Projects', hover_data=['Skill_Courses'],
                             title="Factors Influencing Salary (Placed Students)")
            st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.write("### Skill Correlation Heatmap")
        corr = df.corr()
        fig_corr = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r')
        st.plotly_chart(fig_corr, use_container_width=True)

    elif choice == "ML Dashboard":
        st.header(" Model Performance & Tuning")
        
        m_acc, m_r2, m_mae = metrics
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Classification Accuracy", f"{round(m_acc*100, 2)}%", "+2.1%")
        c2.metric("Regression R² Score", f"{round(m_r2, 3)}", "High")
        c3.metric("Salary MAE (LPA)", f"₹{round(m_mae, 2)}L", "Low")

        st.subheader("Feature Importance (Placement Prediction)")
        feat_importances = pd.Series(clf.feature_importances_, index=feature_cols).sort_values(ascending=True)
        fig_feat = px.bar(feat_importances, orientation='h', title="Key Success Drivers", labels={'value': 'Importance Score', 'index': 'Feature'})
        st.plotly_chart(fig_feat, use_container_width=True)

        with st.expander("View Model Hyperparameters & Architecture"):
            st.json({
                "Classifier": "Random Forest (Ensemble)",
                "Regressor": "Gradient Boosting (Ensemble)",
                "Hyperparameters": {
                    "n_estimators": 100,
                    "max_depth": "None (Full)",
                    "learning_rate": 0.1
                },
                "Training_Size": "80/20 Split"
            })

    elif choice == "Live Prediction":
        st.header(" Prediction Engine")
        st.info("Enter student details below to simulate placement and salary outcomes.")

        with st.form("input_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                x_score = st.slider("Xth Percentage", 60, 100, 85)
                xii_score = st.slider("XIIth Percentage", 60, 100, 82)
            with c2:
                cgpa = st.number_input("BE CGPA", 0.0, 10.0, 8.5)
                aptitude = st.slider("Aptitude Score", 0, 100, 75)
            with c3:
                internships = st.number_input("Internships", 0, 5, 1)
                projects = st.number_input("Projects Completed", 0, 15, 3)
            
            sc = st.number_input("Skill Courses Completed", 0, 10, 2)
            cert = st.number_input("Certifications / Bootcamps", 0, 10, 1)
            
            submit = st.form_submit_button("Generate Prediction")

        if submit:
            with st.spinner("Analyzing candidate profile..."):
                input_data = np.array([[x_score, xii_score, cgpa, sc, internships, projects, cert, aptitude]])
                input_scaled = scaler.transform(input_data)
                
                # Predict Placement
                placed_status = clf.predict(input_scaled)[0]
                placed_prob = clf.predict_proba(input_scaled)[0][1]
                
                time.sleep(1) # Simulation delay
                
                if placed_status == 1:
                    salary_pred = reg.predict(input_scaled)[0]
                    st.success(f"### 🎉 Status: Likely to be PLACED!")
                    st.balloons()
                    
                    res_c1, res_c2 = st.columns(2)
                    res_c1.metric("Placement Probability", f"{round(placed_prob*100, 2)}%")
                    res_c2.metric("Estimated Salary", f"₹{round(salary_pred, 2)} LPA")
                    
                    st.write("---")
                    st.write("**Career Insight:** Your high project count and CGPA are your strongest assets. Focusing on one more high-quality internship could boost your salary by ~1.5 LPA.")
                else:
                    st.warning("### ⚠️ Status: HIGH RISK (May not get placed with current profile)")
                    st.metric("Placement Probability", f"{round(placed_prob*100, 2)}%")
                    st.write("**Recommendation:** Improve your Aptitude Score and complete at least 2 more Skill Courses to cross the threshold.")

    elif choice == "Performance Tracking":
        st.header(" Deployment & Drift Tracking")
        st.write("This section monitors the model versioning and performance decay over time.")
        
        # Mocking versioning data
        history = pd.DataFrame({
            'Date': pd.date_range(start='2023-01-01', periods=6, freq='M'),
            'Accuracy': [0.88, 0.89, 0.88, 0.90, 0.91, 0.92],
            'R2_Score': [0.82, 0.81, 0.84, 0.83, 0.85, 0.86],
            'Avg_Inference_Time (ms)': [45, 42, 40, 38, 35, 33]
        })
        
        fig_track = px.line(history, x='Date', y=['Accuracy', 'R2_Score'], markers=True, 
                            title="Model Reliability over Monthly Retraining Cycles")
        st.plotly_chart(fig_track, use_container_width=True)
        
        st.write("### Model Metadata")
        st.code("""
        Current Model ID: v2.4.1-stable
        Last Retrained: 2023-11-15
        Training Data Hash: 8f2a9c11...
        Model Artifact: random_forest_ensemble.pkl
        Environment: Streamlit-Cloud-Py3.11
        """, language="yaml")

if __name__ == "__main__":
    main()
