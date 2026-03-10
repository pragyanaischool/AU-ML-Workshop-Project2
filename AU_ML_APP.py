import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score, f1_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import time

# --- Page Config ---
st.set_page_config(page_title="Advanced Skill-Based Career AI", layout="wide", page_icon="🎓")

# --- Custom Styling ---
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .stMetric { background-color: white; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .status-box { padding: 20px; border-radius: 10px; margin-bottom: 20px; }
    </style>
    """, unsafe_allow_html=True)

# --- 1. Enhanced Synthetic Data Generation ---
@st.cache_data
def generate_large_dataset(n_samples=5000):
    np.random.seed(42)
    data = {
        'Xth_Score': np.random.uniform(55, 98, n_samples),
        'XIIth_Score': np.random.uniform(55, 98, n_samples),
        'BE_CGPA': np.random.uniform(5.5, 9.9, n_samples),
        'Skill_Courses': np.random.randint(0, 12, n_samples),
        'Internships': np.random.randint(0, 6, n_samples),
        'Projects': np.random.randint(1, 15, n_samples),
        'Certifications': np.random.randint(0, 10, n_samples),
        'Aptitude_Score': np.random.uniform(40, 100, n_samples),
        'Soft_Skills_Rating': np.random.uniform(1, 5, n_samples)
    }
    df = pd.DataFrame(data)
    
    # Complex Placement Logic
    noise = np.random.normal(0, 0.1, n_samples)
    score = (
        df['BE_CGPA'] * 0.35 + 
        df['Internships'] * 1.5 + 
        df['Skill_Courses'] * 0.4 + 
        df['Projects'] * 0.5 + 
        (df['Aptitude_Score']/20) + 
        df['Soft_Skills_Rating']
    )
    df['Placed'] = (score + noise > 12).astype(int)
    
    # Salary Logic (LPA)
    # Base is 3.5L; highly dependent on specific skills and CGPA
    salary = (
        3.5 + 
        (df['BE_CGPA'] - 6).clip(0) * 1.8 + 
        df['Internships'] * 2.5 + 
        df['Projects'] * 0.9 + 
        df['Certifications'] * 0.6 + 
        np.random.normal(0, 0.7, n_samples)
    )
    df['Salary_LPA'] = np.where(df['Placed'] == 1, salary, 0)
    
    return df

# --- 2. Advanced Model Training with Hyperparameter Tuning ---
def train_complex_models(df, run_tuning=False):
    X = df.drop(['Placed', 'Salary_LPA'], axis=1)
    y_class = df['Placed']
    
    # For Regression, only use the placed subset
    df_placed = df[df['Placed'] == 1]
    X_reg = df_placed.drop(['Placed', 'Salary_LPA'], axis=1)
    y_reg = df_placed['Salary_LPA']

    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_class, test_size=0.2, random_state=42)
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_c_s = scaler.fit_transform(X_train_c)
    X_test_c_s = scaler.transform(X_test_c)
    X_train_r_s = scaler.fit_transform(X_train_r)
    X_test_r_s = scaler.transform(X_test_r)

    results = {}

    # --- Classification Section ---
    if run_tuning:
        # Tuning Random Forest
        rf_params = {'n_estimators': [50, 100], 'max_depth': [None, 10, 20]}
        clf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=3)
        clf_grid.fit(X_train_c_s, y_train_c)
        best_clf = clf_grid.best_estimator_
    else:
        best_clf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train_c_s, y_train_c)

    # Secondary Classifier for comparison
    log_reg = LogisticRegression().fit(X_train_c_s, y_train_c)
    
    # --- Regression Section ---
    if run_tuning:
        gb_params = {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1]}
        reg_grid = GridSearchCV(GradientBoostingRegressor(random_state=42), gb_params, cv=3)
        reg_grid.fit(X_train_r_s, y_train_r)
        best_reg = reg_grid.best_estimator_
    else:
        best_reg = GradientBoostingRegressor(n_estimators=100, random_state=42).fit(X_train_r_s, y_train_r)

    # Secondary Regressor
    ridge_reg = Ridge().fit(X_train_r_s, y_train_r)

    # Performance Metrics
    metrics = {
        'clf_acc': accuracy_score(y_test_c, best_clf.predict(X_test_c_s)),
        'clf_f1': f1_score(y_test_c, best_clf.predict(X_test_c_s)),
        'reg_r2': r2_score(y_test_r, best_reg.predict(X_test_r_s)),
        'reg_mae': mean_absolute_error(y_test_r, best_reg.predict(X_test_r_s)),
        'comparison': {
            'LogReg_Acc': accuracy_score(y_test_c, log_reg.predict(X_test_c_s)),
            'Ridge_R2': r2_score(y_test_r, ridge_reg.predict(X_test_r_s))
        }
    }

    return best_clf, best_reg, scaler, metrics, X.columns

# --- Main App ---
def main():
    st.title(" PragyanAI - Skill-Based Talent Intelligence Platform")
    
    # Sidebar Setup
    st.sidebar.header("Control Panel")
    data_size = st.sidebar.slider("Synthetic Data Samples", 1000, 10000, 5000)
    run_tuning = st.sidebar.checkbox("Enable Hyperparameter Tuning", value=False)
    
    df = generate_large_dataset(data_size)
    clf, reg, scaler, metrics, cols = train_complex_models(df, run_tuning)

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Analytics", "⚙️ Model Benchmarking", "🔮 Live Prediction", "📈 Tracking"])

    with tab1:
        st.header("Exploratory Data Insights")
        c1, c2, c3 = st.columns(3)
        c1.metric("Dataset Size", f"{len(df)} rows")
        c2.metric("Avg. CGPA", f"{round(df['BE_CGPA'].mean(), 2)}")
        c3.metric("Placement %", f"{round(df['Placed'].mean()*100, 1)}%")

        st.subheader("Salary Determinants")
        fig_scatter = px.scatter(df[df['Placed']==1], x="BE_CGPA", y="Salary_LPA", 
                                 color="Internships", size="Projects",
                                 title="CGPA vs Salary (Colored by Internships)")
        st.plotly_chart(fig_scatter, use_container_width=True)

        st.subheader("Data Distribution")
        st.dataframe(df.describe().T, use_container_width=True)

    with tab2:
        st.header("Algorithms & Performance")
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Classification (Placed vs Not)")
            perf_df = pd.DataFrame({
                'Model': ['Random Forest (Tuned)', 'Logistic Regression'],
                'Accuracy': [metrics['clf_acc'], metrics['comparison']['LogReg_Acc']]
            })
            st.table(perf_df)
            
            # Feature Importance
            fi = pd.DataFrame({'Feature': cols, 'Importance': clf.feature_importances_}).sort_values('Importance')
            st.plotly_chart(px.bar(fi, x='Importance', y='Feature', orientation='h', title="RF Feature Importance"), use_container_width=True)

        with col_b:
            st.subheader("Regression (Salary Prediction)")
            reg_perf = pd.DataFrame({
                'Model': ['Gradient Boosting', 'Ridge Regression'],
                'R2 Score': [metrics['reg_r2'], metrics['comparison']['Ridge_R2']]
            })
            st.table(reg_perf)
            st.info(f"Mean Absolute Error: ₹{round(metrics['reg_mae'], 2)} LPA")

    with tab3:
        st.header("Predictive Simulator")
        with st.form("user_input"):
            c1, c2 = st.columns(2)
            with c1:
                x_val = st.slider("Xth Score (%)", 50, 100, 80)
                xii_val = st.slider("XIIth Score (%)", 50, 100, 80)
                cgpa_val = st.number_input("Current CGPA", 0.0, 10.0, 8.0)
                apt_val = st.slider("Aptitude Score", 0, 100, 70)
            with c2:
                courses = st.number_input("Skill Courses", 0, 20, 3)
                interns = st.number_input("Internships", 0, 10, 1)
                projs = st.number_input("Total Projects", 0, 30, 4)
                certs = st.number_input("Certifications", 0, 15, 2)
                soft = st.slider("Soft Skills Rating (1-5)", 1.0, 5.0, 3.5)
            
            btn = st.form_submit_button("Run AI Assessment")

        if btn:
            input_arr = np.array([[x_val, xii_val, cgpa_val, courses, interns, projs, certs, apt_val, soft]])
            input_scaled = scaler.transform(input_arr)
            
            is_placed = clf.predict(input_scaled)[0]
            prob = clf.predict_proba(input_scaled)[0][1]
            
            if is_placed == 1:
                sal = reg.predict(input_scaled)[0]
                st.success(f"### Result: ✅ Candidate is likely to be PLACED")
                st.metric("Predicted Salary", f"₹{round(sal, 2)} LPA")
                st.progress(prob)
                st.caption(f"Confidence Level: {round(prob*100, 2)}%")
            else:
                st.error("### Result: ❌ Placement Probability is Low")
                st.metric("Probability", f"{round(prob*100, 2)}%")
                st.write("**Strategy:** Focus on increasing projects and internships to improve outcomes.")

    with tab4:
        st.header("Model Tracking & Health")
        st.write("Simulated drift and monitoring data.")
        drift_data = pd.DataFrame({
            'Batch': ['Jan', 'Feb', 'Mar', 'Apr', 'May'],
            'Model Accuracy': [0.92, 0.91, 0.93, 0.89, 0.90],
            'Data Variance': [0.05, 0.06, 0.04, 0.08, 0.07]
        })
        st.line_chart(drift_data.set_index('Batch'))

if __name__ == "__main__":
    main()
