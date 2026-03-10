import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor, 
    GradientBoostingRegressor, GradientBoostingClassifier,
    AdaBoostClassifier, AdaBoostRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
    VotingClassifier, VotingRegressor
)
from sklearn.linear_model import (
    LogisticRegression, Ridge, Lasso, ElasticNet, 
    Perceptron, PassiveAggressiveClassifier, SGDClassifier
)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score, f1_score
from sklearn.preprocessing import StandardScaler
import time

# --- Page Config ---
st.set_page_config(page_title="Ultra Career AI: 15+ Model Intelligence", layout="wide", page_icon="🚀")

# --- Custom Styling ---
st.markdown("""
    <style>
    .main { background-color: #f4f7f9; }
    .stMetric { background-color: #ffffff; padding: 20px; border-radius: 12px; border: 1px solid #e0e0e0; }
    .prediction-card { padding: 25px; border-radius: 15px; margin-bottom: 25px; border-left: 8px solid #007bff; background: white; }
    </style>
    """, unsafe_allow_html=True)

# --- 1. Massive Synthetic Data Generation ---
@st.cache_data
def generate_robust_dataset(n_samples=7000):
    np.random.seed(42)
    data = {
        'Xth_Score': np.random.uniform(50, 100, n_samples),
        'XIIth_Score': np.random.uniform(50, 100, n_samples),
        'BE_CGPA': np.random.uniform(5.0, 10.0, n_samples),
        'Skill_Courses': np.random.randint(0, 15, n_samples),
        'Internships': np.random.randint(0, 8, n_samples),
        'Projects': np.random.randint(1, 20, n_samples),
        'Certifications': np.random.randint(0, 12, n_samples),
        'Aptitude_Score': np.random.uniform(30, 100, n_samples),
        'Soft_Skills_Rating': np.random.uniform(1, 5, n_samples),
        'Hackathon_Wins': np.random.randint(0, 5, n_samples)
    }
    df = pd.DataFrame(data)
    
    # Placement Logic (Probability based)
    p_score = (
        df['BE_CGPA'] * 0.4 + 
        df['Internships'] * 2.0 + 
        df['Projects'] * 0.6 + 
        df['Hackathon_Wins'] * 1.5 +
        (df['Aptitude_Score'] / 10) + 
        df['Soft_Skills_Rating'] * 2
    )
    df['Placed'] = (p_score + np.random.normal(0, 1.5, n_samples) > 18).astype(int)
    
    # Salary Logic (LPA)
    s_score = (
        4.0 + 
        (df['BE_CGPA'] - 5) * 2.2 + 
        df['Internships'] * 3.1 + 
        df['Projects'] * 1.1 + 
        df['Hackathon_Wins'] * 2.5 +
        df['Skill_Courses'] * 0.4 +
        np.random.normal(0, 1.0, n_samples)
    )
    df['Salary_LPA'] = np.where(df['Placed'] == 1, s_score.clip(3.5, 50.0), 0)
    
    return df

# --- 2. Training Pipeline with 15+ Algorithms ---
def train_multi_model_pipeline(df):
    # Features & Targets
    X = df.drop(['Placed', 'Salary_LPA'], axis=1)
    y_c = df['Placed']
    
    df_p = df[df['Placed'] == 1]
    X_r = df_p.drop(['Placed', 'Salary_LPA'], axis=1)
    y_r = df_p['Salary_LPA']

    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_c, test_size=0.2, random_state=42)
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_r, y_r, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_c_s = scaler.fit_transform(X_train_c)
    X_test_c_s = scaler.transform(X_test_c)
    X_train_r_s = scaler.fit_transform(X_train_r)
    X_test_r_s = scaler.transform(X_test_r)

    # --- CLASSIFICATION MODELS (8 Models) ---
    c_models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "SVC": SVC(probability=True),
        "Extra Trees": ExtraTreesClassifier(n_estimators=100),
        "AdaBoost": AdaBoostClassifier(),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier()
    }
    
    # Ensemble Classifier
    v_clf = VotingClassifier(estimators=[
        ('rf', c_models["Random Forest"]), 
        ('gb', c_models["Gradient Boosting"]), 
        ('et', c_models["Extra Trees"])
    ], voting='soft')
    c_models["Ensemble (Voting)"] = v_clf

    # --- REGRESSION MODELS (8 Models) ---
    r_models = {
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "Ridge": Ridge(),
        "Lasso": Lasso(),
        "Extra Trees": ExtraTreesRegressor(n_estimators=100),
        "AdaBoost": AdaBoostRegressor(),
        "KNN": KNeighborsRegressor(n_neighbors=5),
        "SVR": SVR()
    }

    # Ensemble Regressor
    v_reg = VotingRegressor(estimators=[
        ('rf', r_models["Random Forest"]), 
        ('gb', r_models["Gradient Boosting"]), 
        ('et', r_models["Extra Trees"])
    ])
    r_models["Ensemble (Voting)"] = v_reg

    # Training logic (Fitting best performers for speed in this demo)
    # In a real app, we'd loop and score all; here we fit the Ensembles for final use
    v_clf.fit(X_train_c_s, y_train_c)
    v_reg.fit(X_train_r_s, y_train_r)
    
    # Store performance for all
    c_scores = {}
    for name, m in c_models.items():
        m.fit(X_train_c_s, y_train_c)
        c_scores[name] = accuracy_score(y_test_c, m.predict(X_test_c_s))
        
    r_scores = {}
    for name, m in r_models.items():
        m.fit(X_train_r_s, y_train_r)
        r_scores[name] = r2_score(y_test_r, m.predict(X_test_r_s))

    return v_clf, v_reg, scaler, c_scores, r_scores, X.columns

# --- UI Setup ---
def main():
    st.title(" PragyanAI - Institutional Grade Career Intelligence")
    st.markdown("### Benchmarking 16+ Algorithms for Placement & Salary Forecasting")

    df = generate_robust_dataset(8000)
    best_c, best_r, scaler, c_metrics, r_metrics, feature_cols = train_multi_model_pipeline(df)

    tabs = st.tabs([" Analytics Dashboard", " Algorithm Benchmark", " Predictive Simulator", " Detailed Data"])

    with tabs[0]:
        st.header("Talent Pool Analytics")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Students Analyzed", f"{len(df)}")
        m2.metric("Placement Rate", f"{round(df['Placed'].mean()*100, 1)}%")
        m3.metric("Avg Salary", f"₹{round(df[df['Placed']==1]['Salary_LPA'].mean(), 1)}L")
        m4.metric("Top Salary", f"₹{round(df['Salary_LPA'].max(), 1)}L")

        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(df[df['Placed']==1], x="Salary_LPA", nbins=30, title="Salary Distribution (Placed Only)", color_discrete_sequence=['#2E86C1'])
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig2 = px.box(df, x="Placed", y="BE_CGPA", title="CGPA impact on Placement Status", color="Placed")
            st.plotly_chart(fig2, use_container_width=True)

    with tabs[1]:
        st.header("Multi-Model Benchmarking")
        col_c, col_r = st.columns(2)
        
        with col_c:
            st.subheader("Classification Algorithms (Placement)")
            c_perf = pd.DataFrame(list(c_metrics.items()), columns=['Algorithm', 'Accuracy']).sort_values('Accuracy', ascending=False)
            st.dataframe(c_perf.style.highlight_max(axis=0, color='#D4EFDF'))
            st.plotly_chart(px.bar(c_perf, x='Accuracy', y='Algorithm', orientation='h', title="Classification Leaderboard"), use_container_width=True)

        with col_r:
            st.subheader("Regression Algorithms (Salary)")
            r_perf = pd.DataFrame(list(r_metrics.items()), columns=['Algorithm', 'R2 Score']).sort_values('R2 Score', ascending=False)
            st.dataframe(r_perf.style.highlight_max(axis=0, color='#D4EFDF'))
            st.plotly_chart(px.bar(r_perf, x='R2 Score', y='Algorithm', orientation='h', title="Regression Leaderboard"), use_container_width=True)

    with tabs[2]:
        st.header("Dual-Model Predictive Simulator")
        st.markdown("Using **Ensemble Voting Models** (RF + GB + ExtraTrees) for maximum stability.")
        
        with st.form("input_form"):
            row1 = st.columns(3)
            row2 = st.columns(3)
            row3 = st.columns(4)
            
            with row1[0]: x_val = st.slider("Xth Score (%)", 40, 100, 85)
            with row1[1]: xii_val = st.slider("XIIth Score (%)", 40, 100, 82)
            with row1[2]: cgpa_val = st.number_input("BE CGPA", 0.0, 10.0, 8.4)
            
            with row2[0]: interns = st.number_input("Internships", 0, 10, 2)
            with row2[1]: projs = st.number_input("Major Projects", 0, 20, 4)
            with row2[2]: courses = st.number_input("Skill Courses", 0, 15, 3)
            
            with row3[0]: certs = st.number_input("Certificates", 0, 10, 2)
            with row3[1]: apt = st.slider("Aptitude Score", 0, 100, 78)
            with row3[2]: soft = st.slider("Soft Skills (1-5)", 1.0, 5.0, 4.0)
            with row3[3]: hack = st.number_input("Hackathon Wins", 0, 5, 1)
            
            submit = st.form_submit_button(" Run Multi-Model Analysis")

        if submit:
            input_data = np.array([[x_val, xii_val, cgpa_val, courses, interns, projs, certs, apt, soft, hack]])
            input_scaled = scaler.transform(input_data)
            
            # 1. Placement Model
            placement_res = best_c.predict(input_scaled)[0]
            placement_prob = best_c.predict_proba(input_scaled)[0][1]
            
            # 2. Salary Model
            salary_res = best_r.predict(input_scaled)[0]
            
            st.markdown("---")
            res_c1, res_c2 = st.columns(2)
            
            with res_c1:
                st.markdown(f"<div class='prediction-card'><h3>Placement Status</h3>", unsafe_allow_html=True)
                if placement_res == 1:
                    st.success("✅ LIKELY TO BE PLACED")
                    st.metric("Confidence Level", f"{round(placement_prob*100, 2)}%")
                else:
                    st.error("❌ PLACEMENT UNLIKELY")
                    st.metric("Probability", f"{round(placement_prob*100, 2)}%")
                st.markdown("</div>", unsafe_allow_html=True)
                
            with res_c2:
                st.markdown(f"<div class='prediction-card'><h3>Salary Projection</h3>", unsafe_allow_html=True)
                if placement_res == 1:
                    st.title(f"₹{round(salary_res, 2)} LPA")
                    st.caption("Based on Skill-weighted Ensemble Regression")
                else:
                    st.title("₹0.00 LPA")
                    st.info("Salary is only calculated for predicted placements.")
                st.markdown("</div>", unsafe_allow_html=True)

    with tabs[3]:
        st.header("Raw Dataset Explorer")
        st.dataframe(df, use_container_width=True)

if __name__ == "__main__":
    main()
