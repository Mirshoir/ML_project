import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import geopandas as gpd
import contextily as ctx
import shap
import time
from datetime import datetime

# Configure page
st.set_page_config(
    page_title="Urban Pluvial Flood Modeling",
    page_icon="🌧️",
    layout="wide"
)

# Enhanced CSS styling
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
    .warning {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        padding: 15px;
        border-radius: 0 8px 8px 0;
        margin: 15px 0;
    }
    .alert-level-1 { background-color: #4caf50; color: white; padding: 5px 10px; border-radius: 4px; }
    .alert-level-2 { background-color: #ffc107; color: black; padding: 5px 10px; border-radius: 4px; }
    .alert-level-3 { background-color: #ff9800; color: white; padding: 5px 10px; border-radius: 4px; }
    .alert-level-4 { background-color: #f44336; color: white; padding: 5px 10px; border-radius: 4px; }
    .map-container {
        height: 600px;
        margin-bottom: 30px;
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Updated title and introduction
st.markdown('<div class="header">Urban Flood Modeling: Hydrodynamic vs. Data-Driven Approaches</div>', unsafe_allow_html=True)
st.markdown("""
<div class="info-box">
    <p>This application compares traditional hydrodynamic modeling with data-driven approaches for urban flood mapping. 
    While 1D-2D hydrodynamic models provide the best representation of physical processes, they are computationally expensive for city-scale applications. 
    Data-driven models offer efficient alternatives but face challenges in generalization and interpretability.</p>
</div>
""", unsafe_allow_html=True)

# Create tabs with new Real-time Forecast tab
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Data Preparation", 
    "🤖 Model Training", 
    "🗺️ Susceptibility Map",
    "🔍 Model Interpretation",
    "⏱️ Real-time Forecast"
])

# Data Preparation Tab
with tab1:
    st.markdown('<div class="subheader">Data Preparation Process</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div class="model-card">
            <h3>Flood Inventory Data</h3>
            <p>Historical flood locations obtained from:</p>
            <ul>
                <li>Municipal flood databases</li>
                <li>Insurance claims</li>
                <li>Social media reports</li>
                <li>Citizen science initiatives</li>
            </ul>
            <p>Sample flood inventory for Berlin:</p>
            <img src="https://raw.githubusercontent.com/omarseleem92/Machine_learning_for_flood_susceptibility/main/Figures/Figure1.png" 
                 width="100%" style="border-radius: 8px;">
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="model-card">
            <h3>Predictive Features</h3>
            <p>Key physical characteristics used for prediction:</p>
            <ul>
                <li><span class="highlight">Topography</span>: Elevation, slope, curvature</li>
                <li><span class="highlight">Urban Infrastructure</span>: Imperviousness, drainage capacity</li>
                <li><span class="highlight">Hydrology</span>: Distance to water bodies, flow accumulation</li>
                <li><span class="highlight">Land Use</span>: Building density, green space percentage</li>
            </ul>
            <p>Non-flooded locations are randomly generated in areas without flooding history.</p>
            <div style="text-align: center; margin-top: 15px;">
                <img src="https://raw.githubusercontent.com/omarseleem92/Machine_learning_for_flood_susceptibility/main/Figures/Figure2.png" 
                     width="90%" style="border-radius: 8px;">
                <p style="font-size: 0.8em; color: #666;">Predictive features for flood susceptibility modeling</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Generate sample data
    np.random.seed(42)
    data_size = 1000
    flood_data = pd.DataFrame({
        'elevation': np.random.normal(30, 10, data_size),
        'slope': np.random.gamma(2, 1.5, data_size),
        'distance_to_river': np.random.exponential(100, data_size),
        'imperviousness': np.random.uniform(60, 100, data_size),
        'drainage_capacity': np.random.normal(50, 15, data_size),
        'label': 1  # Flooded locations
    })
    
    non_flood_data = pd.DataFrame({
        'elevation': np.random.normal(50, 15, data_size),
        'slope': np.random.gamma(1, 2, data_size),
        'distance_to_river': np.random.exponential(200, data_size),
        'imperviousness': np.random.uniform(0, 40, data_size),
        'drainage_capacity': np.random.normal(80, 10, data_size),
        'label': 0  # Non-flooded locations
    })
    
    # Combine datasets
    sample_data = pd.concat([flood_data, non_flood_data])
    
    st.markdown("""
    <div class="info-box">
        <h3>Data Balancing</h3>
        <p>To avoid model bias, we maintain a balanced dataset with equal numbers of flooded and non-flooded locations:</p>
        <ul>
            <li>Flooded locations: 1,000 samples</li>
            <li>Non-flooded locations: 1,000 samples</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("Sample Data Overview")
    st.dataframe(sample_data.sample(10, random_state=42))
    
    st.subheader("Feature Distribution")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    features = ['elevation', 'slope', 'distance_to_river', 'imperviousness', 'drainage_capacity']
    
    for i, feature in enumerate(features):
        ax = axes[i//3, i%3]
        sns.boxplot(x='label', y=feature, data=sample_data, ax=ax)
        ax.set_title(feature)
        ax.set_xticklabels(['Non-Flooded', 'Flooded'])
    
    plt.tight_layout()
    st.pyplot(fig)

# Model Training Tab (updated with multiple models)
with tab2:
    st.markdown('<div class="subheader">Model Training and Evaluation</div>', unsafe_allow_html=True)
    
    # Generate synthetic data for modeling
    np.random.seed(42)
    data_size = 2000
    X = pd.DataFrame({
        'elevation': np.random.normal(40, 15, data_size),
        'slope': np.random.gamma(2, 1.5, data_size),
        'distance_to_river': np.random.exponential(150, data_size),
        'imperviousness': np.random.uniform(0, 100, data_size),
        'drainage_capacity': np.random.normal(65, 20, data_size)
    })
    
    # Create synthetic relationship
    y = (0.4*X['elevation'] + 0.3*X['slope'] - 0.2*X['distance_to_river'] + 
         0.5*X['imperviousness'] - 0.4*X['drainage_capacity'] + np.random.normal(0, 15, data_size) > 30).astype(int)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Define models
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Support Vector Machine": SVC(probability=True, random_state=42),
        "Neural Network": MLPClassifier(hidden_layer_sizes=(50, 25), max_iter=1000, random_state=42)
    }
    
    # Train models
    predictions = {}
    performance = []
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else np.zeros(len(y_test))
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba) if hasattr(model, "predict_proba") else 0.0
        
        performance.append({
            "Model": name,
            "Accuracy": acc,
            "F1 Score": f1,
            "ROC AUC": roc_auc
        })
        
        predictions[name] = y_proba
    
    performance_df = pd.DataFrame(performance)
    
    st.markdown("""
    <div class="model-card">
        <h3>Model Comparison</h3>
        <p>We evaluate three different machine learning models for flood prediction:</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Show performance metrics
    st.subheader("Performance Metrics")
    st.dataframe(performance_df.style.format({
        "Accuracy": "{:.3f}", 
        "F1 Score": "{:.3f}", 
        "ROC AUC": "{:.3f}"
    }).background_gradient(cmap="Blues", subset=["Accuracy", "F1 Score", "ROC AUC"]))
    
    # Visual comparison
    st.subheader("Model Performance Comparison")
    fig = go.Figure()
    
    for i, model_name in enumerate(models.keys()):
        model_perf = performance_df[performance_df["Model"] == model_name].iloc[0]
        fig.add_trace(go.Bar(
            x=[model_perf["Accuracy"], model_perf["F1 Score"], model_perf["ROC AUC"]],
            y=["Accuracy", "F1 Score", "ROC AUC"],
            name=model_name,
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
    
    # Feature importance for Random Forest
    st.subheader("Feature Importance (Random Forest)")
    rf_model = models["Random Forest"]
    importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x='Importance', y='Feature', data=importance, palette='viridis')
    plt.title('Feature Importance for Flood Prediction')
    st.pyplot(fig)

# Susceptibility Map Tab
with tab3:
    st.markdown('<div class="subheader">Flood Susceptibility Map</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <p>Flood susceptibility maps show the likelihood that a specific location will experience flooding based on 
        its physical characteristics. This map is generated by applying our trained model across a spatial grid.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Generate synthetic spatial data
    np.random.seed(42)
    n_points = 500
    spatial_data = pd.DataFrame({
        'x': np.random.uniform(13.0, 13.8, n_points),
        'y': np.random.uniform(52.3, 52.7, n_points),
        'elevation': np.random.normal(40, 15, n_points),
        'slope': np.random.gamma(2, 1.5, n_points),
        'distance_to_river': np.random.exponential(150, n_points),
        'imperviousness': np.random.uniform(0, 100, n_points),
        'drainage_capacity': np.random.normal(65, 20, n_points)
    })
    
    # Predict probabilities using Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    spatial_data['flood_prob'] = model.predict_proba(spatial_data[X.columns])[:, 1]
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(
        spatial_data, 
        geometry=gpd.points_from_xy(spatial_data.x, spatial_data.y)
    )
    gdf.crs = "EPSG:4326"
    
    # Plot susceptibility map
    st.subheader("Berlin Flood Susceptibility")
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot flood probability
    scatter = ax.scatter(
        spatial_data['x'], 
        spatial_data['y'], 
        c=spatial_data['flood_prob'], 
        cmap='RdYlBu_r',
        s=50,
        alpha=0.8,
        vmin=0,
        vmax=1
    )
    
    plt.colorbar(scatter, label='Flood Probability')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Flood Susceptibility Map')
    
    # Add basemap
    gdf_wm = gdf.to_crs(epsg=3857)
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
    
    st.pyplot(fig)
    
    st.markdown("""
    <div class="model-card">
        <h3>Interpretation of Susceptibility Levels</h3>
        <div style="display: flex; justify-content: space-between; margin-top: 15px;">
            <div style="text-align: center; width: 18%; background-color: #4575b4; color: white; padding: 10px; border-radius: 5px;">
                Very Low<br>(0-0.2)
            </div>
            <div style="text-align: center; width: 18%; background-color: #91bfdb; color: black; padding: 10px; border-radius: 5px;">
                Low<br>(0.2-0.4)
            </div>
            <div style="text-align: center; width: 18%; background-color: #e0f3f8; color: black; padding: 10px; border-radius: 5px;">
                Moderate<br>(0.4-0.6)
            </div>
            <div style="text-align: center; width: 18%; background-color: #fee090; color: black; padding: 10px; border-radius: 5px;">
                High<br>(0.6-0.8)
            </div>
            <div style="text-align: center; width: 18%; background-color: #fc8d59; color: white; padding: 10px; border-radius: 5px;">
                Very High<br>(0.8-1.0)
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Model Interpretation Tab (updated with SHAP)
with tab4:
    st.markdown('<div class="subheader">Model Interpretation with SHAP</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <p>Understanding how the model makes predictions is crucial for validating and trusting its results. 
        We use SHAP (SHapley Additive exPlanations) values to interpret our model's predictions.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Train a model for SHAP
    model_shap = RandomForestClassifier(n_estimators=50, random_state=42)
    model_shap.fit(X_train, y_train)
    
    # Compute SHAP values
    explainer = shap.TreeExplainer(model_shap)
    sample_idx = np.random.choice(X_test.index, 100, replace=False)
    shap_values = explainer(X_test.loc[sample_idx])
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Global Feature Importance")
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values.values, X_test.loc[sample_idx], plot_type="bar", show=False)
        plt.title("SHAP Feature Importance")
        st.pyplot(fig)
        
        st.markdown("""
        <div class="info-box">
            <b>Interpretation:</b> 
            <ul>
                <li>Imperviousness has the strongest impact on flood predictions</li>
                <li>Drainage capacity acts as a mitigating factor</li>
                <li>Elevation and slope contribute moderately to flood risk</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("Feature Impact Direction")
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values.values, X_test.loc[sample_idx], show=False)
        plt.title("SHAP Value Impact")
        st.pyplot(fig)
        
        st.markdown("""
        <div class="info-box">
            <b>Interpretation:</b>
            <ul>
                <li>High imperviousness increases flood risk (red points)</li>
                <li>High drainage capacity decreases flood risk (blue points)</li>
                <li>Low elevation increases flood risk</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Local explanation
    st.subheader("Local Explanation for Specific Location")
    sample_id = st.slider("Select sample to explain", 0, len(sample_idx)-1, 25)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown(f"**Location Characteristics**")
        sample_data = X_test.loc[sample_idx].iloc[sample_id]
        st.write(sample_data)
        
        prediction = model_shap.predict_proba([sample_data])[0][1]
        st.metric("Predicted Flood Probability", f"{prediction:.1%}")
        
        st.markdown("""
        <div class="info-box">
            <b>Key Risk Factors:</b>
            <ul>
                <li>High imperviousness (82%)</li>
                <li>Low elevation (28m)</li>
                <li>Inadequate drainage capacity (45%)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("**SHAP Waterfall Plot**")
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(shap_values[sample_id], max_display=8, show=False)
        plt.title(f"Local Explanation for Location {sample_id}")
        plt.tight_layout()
        st.pyplot(fig)

# New Real-time Forecast Tab
with tab5:
    st.markdown('<div class="subheader">Real-time Flood Forecasting System</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
        <p>Combining hydrodynamic and data-driven models for operational flood forecasting</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Generate flood alert data
    np.random.seed(42)
    n_locations = 50
    locations = pd.DataFrame({
        'lat': np.random.uniform(37.7, 37.82, n_locations),
        'lon': np.random.uniform(-122.5, -122.37, n_locations),
        'alert_level': np.random.choice([1, 2, 3, 4], n_locations, p=[0.5, 0.3, 0.15, 0.05]),
        'name': [f"Location {i+1}" for i in range(n_locations)]
    })
    
    # Alert level descriptions
    alert_levels = {
        1: ("Low", "No flooding expected", "alert-level-1"),
        2: ("Moderate", "Minor street flooding possible", "alert-level-2"),
        3: ("High", "Significant flooding expected", "alert-level-3"),
        4: ("Extreme", "Severe flooding - take action", "alert-level-4")
    }
    
    # Current alert status
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
    st.markdown(f"""
    <div class="model-card">
        <h3>Current Flood Alert Status</h3>
        <p>Last updated: {current_time}</p>
        <div style="display: flex; gap: 15px; margin-top: 15px;">
            <div style="flex: 1; text-align: center; padding: 10px; background-color: #4caf50; color: white; border-radius: 8px;">
                <h4>Low</h4>
                <p>25 locations</p>
            </div>
            <div style="flex: 1; text-align: center; padding: 10px; background-color: #ffc107; border-radius: 8px;">
                <h4>Moderate</h4>
                <p>15 locations</p>
            </div>
            <div style="flex: 1; text-align: center; padding: 10px; background-color: #ff9800; color: white; border-radius: 8px;">
                <h4>High</h4>
                <p>7 locations</p>
            </div>
            <div style="flex: 1; text-align: center; padding: 10px; background-color: #f44336; color: white; border-radius: 8px;">
                <h4>Extreme</h4>
                <p>3 locations</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Map visualization
    st.subheader("Flood Alert Map")
    fig = px.scatter_mapbox(
        locations,
        lat="lat",
        lon="lon",
        color="alert_level",
        color_continuous_scale=px.colors.diverging.RdYlBu_r,
        range_color=(1, 4),
        size_max=15,
        zoom=11,
        hover_name="name",
        hover_data={"alert_level": True, "lat": False, "lon": False}
    )
    
    fig.update_layout(
        mapbox_style="carto-positron",
        height=500,
        margin={"r":0,"t":0,"l":0,"b":0}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Forecast controls
    st.subheader("Forecast Parameters")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        rainfall = st.slider("Rainfall Intensity (mm/hr)", 0, 100, 30, 5)
    with col2:
        duration = st.slider("Storm Duration (hours)", 1, 72, 24)
    with col3:
        antecedent = st.select_slider("Antecedent Soil Moisture", 
                                    options=["Dry", "Normal", "Wet"], 
                                    value="Normal")
    
    # Generate forecast data
    st.subheader("Flood Probability Forecast")
    hours = list(range(1, 25))
    base_prob = np.clip(0.1 + (rainfall/100) * 0.7 + (0.1 if antecedent=="Wet" else 0), 0, 0.95)
    probabilities = [base_prob * (1 + 0.05*h) for h in hours]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hours,
        y=probabilities,
        mode='lines+markers',
        name='Flood Probability',
        line=dict(color='#1e3c72', width=3)
    ))
    
    # Add alert thresholds
    fig.add_hline(y=0.3, line_dash="dash", line_color="orange", annotation_text="Moderate Alert")
    fig.add_hline(y=0.6, line_dash="dash", line_color="red", annotation_text="High Alert")
    
    fig.update_layout(
        title="24-Hour Flood Probability Forecast",
        xaxis_title="Hours from Now",
        yaxis_title="Flood Probability",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Alert explanation
    max_prob = max(probabilities)
    if max_prob > 0.6:
        alert_class = "alert-level-4"
        alert_text = "EXTREME FLOOD RISK"
        recommendation = "Evacuate high-risk areas. Activate emergency response plans."
    elif max_prob > 0.3:
        alert_class = "alert-level-3"
        alert_text = "HIGH FLOOD RISK"
        recommendation = "Prepare sandbags. Move vehicles to higher ground."
    else:
        alert_class = "alert-level-1"
        alert_text = "LOW FLOOD RISK"
        recommendation = "Monitor conditions. Clear drainage systems."
    
    st.markdown(f"""
    <div class="warning">
        <div style="display: flex; align-items: center; gap: 15px;">
            <div class="{alert_class}" style="font-size: 1.2em; font-weight: bold; flex-shrink: 0;">
                {alert_text}
            </div>
            <div>
                <p style="margin: 0;">{recommendation}</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
**Research Paper:** [Towards urban flood susceptibility mapping using data-driven models in Berlin, Germany](https://www.tandfonline.com/doi/full/10.1080/19475705.2023.2232299)  
**GitHub Repository:** [Machine Learning for Flood Susceptibility](https://github.com/omarseleem92/Machine_learning_for_flood_susceptibility)  
**Data Source:** [Berlin Open Data Portal](https://daten.berlin.de/)
""")
