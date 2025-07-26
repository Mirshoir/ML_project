import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score
import shap
import time
import geopandas as gpd
import contextily as ctx
import io

# Configure page
st.set_page_config(
    page_title="Urban Flood Modeling Comparison",
    page_icon="🌧️",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .header {
        color: #1e3c72;
        font-size: 2.5em;
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
    }
    .subheader {
        color: #2a5298;
        font-size: 1.5em;
        margin-top: 30px;
        border-bottom: 2px solid #1e3c72;
        padding-bottom: 5px;
    }
    .info-box {
        background-color: #f0f5ff;
        border-radius: 10px;
        padding: 20px;
        margin: 15px 0;
        border-left: 4px solid #1e3c72;
    }
    .model-card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        border: 1px solid #ddd;
    }
    .highlight {
        background-color: #fff9c4;
        padding: 5px 10px;
        border-radius: 5px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Title and introduction
st.markdown('<div class="header">Urban Flood Modeling: Hydrodynamic vs. Data-Driven Approaches</div>', unsafe_allow_html=True)
st.markdown("""
<div class="info-box">
    <p>This application compares traditional hydrodynamic modeling with data-driven approaches for urban flood mapping. 
    While 1D-2D hydrodynamic models provide the best representation of physical processes, they are computationally expensive for city-scale applications. 
    Data-driven models offer efficient alternatives but face challenges in generalization and interpretability.</p>
</div>
""", unsafe_allow_html=True)

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🏙️ Overview", 
    "🤖 Model Comparison", 
    "🌍 Spatial Transferability",
    "🔍 Model Interpretation"
])

# Overview Tab
with tab1:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="subheader">Hydrodynamic Modeling</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="model-card">
            <h3>1D-2D Hydrodynamic Models</h3>
            <p>Represents physical processes of runoff generation and concentration</p>
            <ul>
                <li><b>Strengths</b>: Physically-based, accurate for small areas</li>
                <li><b>Limitations</b>: Computationally expensive, requires detailed inputs</li>
                <li><b>Applications</b>: Small study areas, detailed flood hazard assessment</li>
            </ul>
            <div style="text-align: center; margin: 15px 0;">
                <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/0d/Hydrodynamic_Modeling_Diagram.svg/1200px-Hydrodynamic_Modeling_Diagram.svg.png" width="100%">
                <p style="font-size: 0.8em; color: #666;">Hydrodynamic modeling workflow (Source: Wikimedia Commons)</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="subheader">Data-Driven Modeling</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="model-card">
            <h3>Machine Learning & Deep Learning</h3>
            <p>Learn relationships between input factors and flood outcomes</p>
            <ul>
                <li><b>Strengths</b>: Computationally efficient, handles complex patterns</li>
                <li><b>Challenges</b>: Black-box nature, limited spatial transferability</li>
                <li><b>Applications</b>: City-scale flood susceptibility mapping</li>
            </ul>
            <div style="text-align: center; margin: 15px 0;">
                <img src="https://miro.medium.com/v2/resize:fit:1400/1*8q0ZJ2xJ9ZJ9ZJ9ZJ9ZJ9Q.png" width="100%">
                <p style="font-size: 0.8em; color: #666;">Data-driven modeling workflow (Source: Medium)</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <h3>Key Research Questions</h3>
        <ol>
            <li>How do deep learning models compare with traditional machine learning for flood susceptibility mapping?</li>
            <li>Can models generalize to areas not included in the training dataset?</li>
            <li>How can we interpret and explain the predictions of black-box models?</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

# Model Comparison Tab
with tab2:
    st.markdown('<div class="subheader">Model Performance Comparison</div>', unsafe_allow_html=True)
    
    # Generate sample data
    np.random.seed(42)
    data_size = 5000
    X = pd.DataFrame({
        'elevation': np.random.normal(50, 15, data_size),
        'slope': np.random.gamma(2, 1.5, data_size),
        'distance_to_river': np.random.exponential(50, data_size),
        'imperviousness': np.random.uniform(0, 100, data_size),
        'drainage_capacity': np.random.normal(70, 20, data_size)
    })
    y = (0.3*X['elevation'] + 0.2*X['slope'] - 0.1*X['distance_to_river'] + 
         0.4*X['imperviousness'] - 0.3*X['drainage_capacity'] + np.random.normal(0, 10, data_size) > 30).astype(int)
    
    # Train/test split
    X_train, X_test = X[:4000], X[4000:]
    y_train, y_test = y[:4000], y[4000:]
    
    # Model training
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100),
        "Support Vector Machine": SVC(probability=True),
        "Neural Network": MLPClassifier(hidden_layer_sizes=(50, 25), max_iter=1000)
    }
    
    results = []
    predictions = {}
    
    for name, model in models.items():
        with st.spinner(f"Training {name}..."):
            start_time = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - start_time
            
            y_pred = model.predict(X_test)
            proba = model.predict_proba(X_test)[:, 1]
            
            predictions[name] = proba
            
            results.append({
                "Model": name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "F1 Score": f1_score(y_test, y_pred),
                "Training Time (s)": train_time,
                "Parameters": model.get_params()
            })
    
    # Show results
    results_df = pd.DataFrame(results)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Performance Metrics")
        st.dataframe(results_df[["Model", "Accuracy", "F1 Score", "Training Time (s)"]].style
                     .format({"Accuracy": "{:.3f}", "F1 Score": "{:.3f}", "Training Time (s)": "{:.2f}"})
                     .background_gradient(cmap="Blues", subset=["Accuracy", "F1 Score"])
                     .background_gradient(cmap="Reds", subset=["Training Time (s)"]))
    
    with col2:
        st.subheader("Comparison Visualization")
        fig = go.Figure()
        
        for i, row in results_df.iterrows():
            fig.add_trace(go.Bar(
                x=[row["Accuracy"], row["F1 Score"]],
                y=["Accuracy", "F1 Score"],
                name=row["Model"],
                orientation='h',
                marker_color=['#1e3c72', '#2a5298', '#3a6ea5'][i]
            ))
        
        fig.update_layout(
            barmode='group',
            title="Model Performance Comparison",
            xaxis_title="Score",
            yaxis_title="Metric",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance
    st.subheader("Feature Importance")
    rf = models["Random Forest"]
    importances = pd.DataFrame({
        "Feature": X.columns,
        "Importance": rf.feature_importances_
    }).sort_values("Importance", ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(x="Importance", y="Feature", data=importances, palette="Blues_d", ax=ax)
    ax.set_title("Random Forest Feature Importance")
    st.pyplot(fig)

# Spatial Transferability Tab
with tab3:
    st.markdown('<div class="subheader">Spatial Transferability Analysis</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
        <p>Evaluating model performance when applied to new geographic areas not included in the training dataset.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Generate spatial data
    np.random.seed(42)
    n_points = 1000
    train_area = pd.DataFrame({
        'x': np.random.uniform(0, 10, n_points),
        'y': np.random.uniform(0, 10, n_points),
        'area': 'Training Area',
        'flood_prob': np.random.beta(2, 5, n_points)
    })
    
    test_area = pd.DataFrame({
        'x': np.random.uniform(8, 18, n_points),
        'y': np.random.uniform(8, 18, n_points),
        'area': 'Test Area',
        'flood_prob': np.random.beta(4, 3, n_points)  # Different distribution
    })
    
    spatial_df = pd.concat([train_area, test_area])
    
    # Performance comparison
    st.subheader("Model Performance in New Areas")
    
    # Simulate performance metrics
    metrics = pd.DataFrame({
        "Model": ["Random Forest", "SVM", "Neural Network"] * 2,
        "Area": ["Training"] * 3 + ["Test"] * 3,
        "Accuracy": [0.92, 0.89, 0.91, 0.75, 0.68, 0.73],
        "F1 Score": [0.90, 0.87, 0.89, 0.72, 0.65, 0.70]
    })
    
    fig = px.bar(metrics, x="Model", y="Accuracy", color="Area", barmode="group",
                 title="Accuracy Comparison: Training vs. Test Areas",
                 color_discrete_sequence=["#1e3c72", "#3a6ea5"])
    st.plotly_chart(fig, use_container_width=True)
    
    # Spatial visualization
    st.subheader("Flood Susceptibility Mapping")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("**Training Area**")
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(train_area['x'], train_area['y'], c=train_area['flood_prob'], 
                            cmap='RdYlBu_r', s=20, alpha=0.7, vmin=0, vmax=1)
        plt.colorbar(scatter, label='Flood Probability')
        ax.set_title("Training Area: Flood Susceptibility")
        ax.set_xlim(0, 18)
        ax.set_ylim(0, 18)
        st.pyplot(fig)
    
    with col2:
        st.markdown("**Test Area**")
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(test_area['x'], test_area['y'], c=test_area['flood_prob'], 
                            cmap='RdYlBu_r', s=20, alpha=0.7, vmin=0, vmax=1)
        plt.colorbar(scatter, label='Flood Probability')
        ax.set_title("Test Area: Flood Susceptibility")
        ax.set_xlim(0, 18)
        ax.set_ylim(0, 18)
        st.pyplot(fig)
    
    # Transferability strategies
    st.subheader("Improving Spatial Transferability")
    st.markdown("""
    <div class="model-card">
        <h3>Strategies for Better Generalization</h3>
        <ul>
            <li><span class="highlight">Domain Adaptation</span>: Techniques to align feature distributions between areas</li>
            <li><span class="highlight">Transfer Learning</span>: Pre-train on large dataset, fine-tune on target area</li>
            <li><span class="highlight">Physics-Informed ML</span>: Incorporate physical constraints into models</li>
            <li><span class="highlight">Ensemble Methods</span>: Combine predictions from multiple models</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Model Interpretation Tab
with tab4:
    st.markdown('<div class="subheader">Interpreting Black-Box Models</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
        <p>Techniques to understand how data-driven models make flood predictions</p>
    </div>
    """, unsafe_allow_html=True)
    
    # SHAP analysis
    st.subheader("SHAP (SHapley Additive exPlanations)")
    
    # Train a model for SHAP
    model = RandomForestClassifier(n_estimators=50)
    model.fit(X_train, y_train)
    
    # Compute SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test.iloc[:100])
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("**Summary Plot**")
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values[1], X_test.iloc[:100], plot_type="dot", show=False)
        st.pyplot(fig)
    
    with col2:
        st.markdown("**Feature Dependence**")
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.dependence_plot("imperviousness", shap_values[1], X_test.iloc[:100], show=False)
        st.pyplot(fig)
    
    # Local explanation
    st.subheader("Local Interpretation for Specific Locations")
    
    sample_idx = st.slider("Select sample to explain", 0, len(X_test)-1, 25)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown(f"**Sample Characteristics**")
        st.dataframe(X_test.iloc[sample_idx:sample_idx+1])
        
        prediction = model.predict_proba(X_test.iloc[sample_idx:sample_idx+1])[0][1]
        actual = y_test.iloc[sample_idx]
        
        st.metric("Predicted Flood Probability", f"{prediction:.2f}")
        st.metric("Actual Status", "Flood" if actual == 1 else "No Flood")
    
    with col2:
        st.markdown("**SHAP Force Plot**")
        fig, ax = plt.subplots(figsize=(10, 4))
        shap.force_plot(
            explainer.expected_value[1], 
            shap_values[1][sample_idx], 
            X_test.iloc[sample_idx],
            matplotlib=True,
            show=False
        )
        st.pyplot(fig)
    
    # Interpretation techniques comparison
    st.subheader("Interpretation Methods Comparison")
    st.markdown("""
    <div class="model-card">
        <table style="width:100%; border-collapse: collapse;">
            <tr style="background-color: #1e3c72; color: white;">
                <th style="padding: 10px;">Method</th>
                <th style="padding: 10px;">Global Interpretation</th>
                <th style="padding: 10px;">Local Interpretation</th>
                <th style="padding: 10px;">Computation Time</th>
                <th style="padding: 10px;">Model Compatibility</th>
            </tr>
            <tr>
                <td style="padding: 10px; border-bottom: 1px solid #ddd;">SHAP</td>
                <td style="padding: 10px; border-bottom: 1px solid #ddd;">✓</td>
                <td style="padding: 10px; border-bottom: 1px solid #ddd;">✓</td>
                <td style="padding: 10px; border-bottom: 1px solid #ddd;">High</td>
                <td style="padding: 10px; border-bottom: 1px solid #ddd;">Tree-based, Neural Nets</td>
            </tr>
            <tr>
                <td style="padding: 10px; border-bottom: 1px solid #ddd;">LIME</td>
                <td style="padding: 10px; border-bottom: 1px solid #ddd;">✗</td>
                <td style="padding: 10px; border-bottom: 1px solid #ddd;">✓</td>
                <td style="padding: 10px; border-bottom: 1px solid #ddd;">Medium</td>
                <td style="padding: 10px; border-bottom: 1px solid #ddd;">All models</td>
            </tr>
            <tr>
                <td style="padding: 10px; border-bottom: 1px solid #ddd;">Partial Dependence</td>
                <td style="padding: 10px; border-bottom: 1px solid #ddd;">✓</td>
                <td style="padding: 10px; border-bottom: 1px solid #ddd;">✗</td>
                <td style="padding: 10px; border-bottom: 1px solid #ddd;">Low</td>
                <td style="padding: 10px; border-bottom: 1px solid #ddd;">All models</td>
            </tr>
            <tr>
                <td style="padding: 10px;">Permutation Importance</td>
                <td style="padding: 10px;">✓</td>
                <td style="padding: 10px;">✗</td>
                <td style="padding: 10px;">High</td>
                <td style="padding: 10px;">All models</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.caption("© 2023 Urban Flood Modeling Research | [Part 1: Data-Driven Model for Mapping Urban Flooding](https://medium.com/hydroinformatics/data-driven-model-for-mapping-urban-flooding-1-d182a1e2dc9)")
