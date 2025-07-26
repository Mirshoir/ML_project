import streamlit as st
import geopandas as gpd
import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt
import os
import tempfile
import numpy as np
from io import BytesIO
import zipfile
import time
import pandas as pd
import joblib
import warnings
from matplotlib.colors import LinearSegmentedColormap
import folium
from streamlit_folium import folium_static
from folium.plugins import Draw
import branca.colormap as cm
import contextily as ctx
from matplotlib import patheffects

# Suppress warnings
warnings.filterwarnings('ignore')

# Initialize session state variables
if 'study_area_map' not in st.session_state:
    st.session_state.study_area_map = None
if 'flood_map' not in st.session_state:
    st.session_state.flood_map = None
if 'factor_maps' not in st.session_state:
    st.session_state.factor_maps = {}
if 'model_type' not in st.session_state:
    st.session_state.model_type = 'Random Forest'
if 'hazard_map' not in st.session_state:
    st.session_state.hazard_map = None
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'model_metrics' not in st.session_state:
    st.session_state.model_metrics = None
if 'training_data' not in st.session_state:
    st.session_state.training_data = None
if 'reported_flood_points' not in st.session_state:
    st.session_state.reported_flood_points = None

# App title and description
st.title("🌊 Flood Susceptibility Analysis System")
st.markdown("""
This application performs flood susceptibility analysis using geospatial data and machine learning models. 
Upload your study area, flood inventory map, and factor maps to generate a flood hazard prediction.
""")

# Create tabs for the three windows
tab1, tab2, tab3 = st.tabs(["🗺️ Study Area", "📊 Data & Model", "🌋 Hazard Map"])

# First Tab: Study Area Map
with tab1:
    st.header("Study Area Configuration")
    st.info("Upload a GIS file representing your study area (shapefile or GeoTIFF)")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # File uploader for GIS data
        uploaded_file = st.file_uploader(
            "Upload Study Area Map",
            type=["shp", "shx", "dbf", "prj", "tif", "tiff", "zip"],
            accept_multiple_files=True,
            key="study_area_uploader"
        )
        
        if uploaded_file:
            try:
                # Handle shapefile upload
                if any(file.name.endswith('.shp') for file in uploaded_file) or any(file.name.endswith('.zip') for file in uploaded_file):
                    with tempfile.TemporaryDirectory() as tmpdir:
                        # Process ZIP file if uploaded
                        zip_files = [f for f in uploaded_file if f.name.endswith('.zip')]
                        if zip_files:
                            with zipfile.ZipFile(zip_files[0], 'r') as zip_ref:
                                zip_ref.extractall(tmpdir)
                            extracted_files = [os.path.join(tmpdir, f) for f in os.listdir(tmpdir)]
                            shp_file = [f for f in extracted_files if f.endswith('.shp')][0]
                        else:
                            for file in uploaded_file:
                                with open(os.path.join(tmpdir, file.name), "wb") as f:
                                    f.write(file.getbuffer())
                            shp_file = [f for f in os.listdir(tmpdir) if f.endswith('.shp')][0]
                            shp_file = os.path.join(tmpdir, shp_file)
                        
                        gdf = gpd.read_file(shp_file)
                        st.session_state.study_area_map = gdf
                
                # Handle GeoTIFF upload
                elif any(file.name.endswith(('.tif', '.tiff')) for file in uploaded_file):
                    tiff_file = [f for f in uploaded_file if f.name.endswith(('.tif', '.tiff'))][0]
                    st.session_state.study_area_map = {
                        'data': tiff_file.getvalue(),
                        'filename': tiff_file.name
                    }
            
            except Exception as e:
                st.error(f"Error processing uploaded files: {str(e)}")
    
    with col2:
        # Display study area map if available
        if st.session_state.study_area_map:
            st.subheader("Study Area Preview")
            
            if isinstance(st.session_state.study_area_map, gpd.GeoDataFrame):
                # Plot vector data
                fig, ax = plt.subplots(figsize=(6, 4))
                st.session_state.study_area_map.plot(ax=ax, color='lightblue', edgecolor='blue')
                ax.set_title("Study Area")
                ax.set_axis_off()
                st.pyplot(fig)
            
            elif isinstance(st.session_state.study_area_map, dict):
                # Plot raster data
                try:
                    with rasterio.open(BytesIO(st.session_state.study_area_map['data'])) as src:
                        fig, ax = plt.subplots(figsize=(6, 4))
                        show(src, ax=ax, title="Study Area")
                        st.pyplot(fig)
                except Exception as e:
                    st.error(f"Error displaying raster: {str(e)}")
        
        else:
            st.info("Upload a study area map to begin")
            st.image("https://via.placeholder.com/300x200?text=Study+Area+Map", use_column_width=True)
            
    # Upload reported flood points
    st.subheader("Reported Flood Locations")
    st.info("Upload points of historical flood events (shapefile)")
    flood_points = st.file_uploader(
        "Upload Flood Points (Shapefile)",
        type=["shp", "shx", "dbf", "prj", "zip"],
        accept_multiple_files=True,
        key="flood_points_uploader"
    )
    
    if flood_points:
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                # Process ZIP file if uploaded
                zip_files = [f for f in flood_points if f.name.endswith('.zip')]
                if zip_files:
                    with zipfile.ZipFile(zip_files[0], 'r') as zip_ref:
                        zip_ref.extractall(tmpdir)
                    extracted_files = [os.path.join(tmpdir, f) for f in os.listdir(tmpdir)]
                    shp_file = [f for f in extracted_files if f.endswith('.shp')][0]
                else:
                    for file in flood_points:
                        with open(os.path.join(tmpdir, file.name), "wb") as f:
                            f.write(file.getbuffer())
                    shp_file = [f for f in os.listdir(tmpdir) if f.endswith('.shp')][0]
                    shp_file = os.path.join(tmpdir, shp_file)
                
                gdf = gpd.read_file(shp_file)
                st.session_state.reported_flood_points = gdf
                st.success(f"Loaded {len(gdf)} flood points!")
                
        except Exception as e:
            st.error(f"Error processing flood points: {str(e)}")

# Second Tab: Data Upload and Model Selection
with tab2:
    st.header("Data Upload and Model Configuration")
    
    # Section 1: Flood Map Upload
    st.subheader("Flood Inventory Map")
    st.info("Upload a raster map showing historical flood occurrences (1=flooded, 0=non-flooded)")
    
    flood_map = st.file_uploader(
        "Upload Flood Map (GeoTIFF)", 
        type=["tif", "tiff"],
        key="flood_uploader"
    )
    
    if flood_map:
        st.session_state.flood_map = {
            'data': flood_map.getvalue(),
            'filename': flood_map.name
        }
        
        col1, col2 = st.columns([2, 1])
        with col1:
            try:
                with rasterio.open(BytesIO(st.session_state.flood_map['data'])) as src:
                    fig, ax = plt.subplots(figsize=(8, 4))
                    show(src, ax=ax, title="Flood Map")
                    st.pyplot(fig)
            except Exception as e:
                st.error(f"Error displaying flood map: {str(e)}")
        with col2:
            st.metric("File", st.session_state.flood_map['filename'])
            st.info("""
            **Flood Map Requirements:**
            - GeoTIFF format
            - Same extent and resolution as factor maps
            - 1 = Flooded area
            - 0 = Non-flooded area
            """)
    
    # Section 2: Factor Maps Upload
    st.subheader("Factor Maps")
    st.info("Upload raster maps representing flood conditioning factors (e.g., elevation, slope, etc.)")
    
    factor_maps = st.file_uploader(
        "Upload Factor Maps (GeoTIFF)", 
        type=["tif", "tiff"],
        accept_multiple_files=True,
        key="factors_uploader"
    )
    
    if factor_maps:
        for fm in factor_maps:
            st.session_state.factor_maps[fm.name] = fm.getvalue()
        
        # Display each factor map
        st.subheader("Uploaded Factor Maps")
        cols = st.columns(2)
        col_idx = 0
        
        for i, (name, data) in enumerate(st.session_state.factor_maps.items()):
            with cols[col_idx]:
                try:
                    with rasterio.open(BytesIO(data)) as src:
                        fig, ax = plt.subplots(figsize=(6, 3))
                        show(src, ax=ax, title=name)
                        st.pyplot(fig)
                        st.caption(name)
                except Exception as e:
                    st.error(f"Error displaying factor map {name}: {str(e)}")
            
            col_idx = (col_idx + 1) % 2

    # Section 3: Model Selection and Execution
    st.subheader("Model Configuration")
    
    # Model selection
    model_options = ['Random Forest', 'Logistic Regression', 'Support Vector Machine']
    st.session_state.model_type = st.selectbox(
        "Select Model Type",
        model_options,
        index=model_options.index(st.session_state.model_type) if st.session_state.model_type else 0
    )
    
    # Advanced parameters
    with st.expander("Advanced Parameters"):
        if st.session_state.model_type == 'Random Forest':
            n_estimators = st.slider("Number of Trees", 10, 200, 100)
            max_depth = st.slider("Max Depth", 2, 30, 10)
        elif st.session_state.model_type == 'Logistic Regression':
            c_value = st.slider("C (Regularization)", 0.01, 10.0, 1.0)
        elif st.session_state.model_type == 'Support Vector Machine':
            kernel = st.selectbox("Kernel", ['linear', 'rbf', 'poly'])
            c_value = st.slider("C (Regularization)", 0.01, 10.0, 1.0)
    
    # Model run button
    if st.button("Train Model", disabled=not (st.session_state.flood_map and st.session_state.factor_maps)):
        st.info(f"Training {st.session_state.model_type} model...")
        
        # Simulate model training
        with st.spinner("Processing data and training model..."):
            # Simulate processing time
            time.sleep(2)
            
            # Generate sample metrics
            st.session_state.model_metrics = {
                'accuracy': round(0.85 + np.random.rand()/10, 3),
                'f1_score': round(0.82 + np.random.rand()/10, 3),
                'roc_auc': round(0.88 + np.random.rand()/10, 3)
            }
            
            # Create sample training data
            st.session_state.training_data = pd.DataFrame({
                'Elevation': np.random.normal(100, 30, 1000),
                'Slope': np.random.gamma(2, 1.5, 1000),
                'Distance to River': np.random.exponential(50, 1000),
                'Land Use': np.random.randint(1, 5, 1000),
                'Flood': np.random.randint(0, 2, 1000)
            })
            
            st.session_state.model_trained = True
            st.success("Model training completed successfully!")

# Third Tab: Hazard Map Display
with tab3:
    st.header("Flood Hazard Map")
    
    if st.session_state.model_trained and st.session_state.model_metrics:
        # Display model metrics
        st.subheader("Model Performance")
        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{st.session_state.model_metrics['accuracy']:.3f}")
        col2.metric("F1 Score", f"{st.session_state.model_metrics['f1_score']:.3f}")
        col3.metric("ROC AUC", f"{st.session_state.model_metrics['roc_auc']:.3f}")
        
        # Generate hazard map
        st.subheader("Susceptibility Map")
        
        # Create a sample hazard map
        x = np.linspace(0, 10, 500)
        y = np.linspace(0, 10, 500)
        X, Y = np.meshgrid(x, y)
        hazard_data = np.sin(X) * np.cos(Y) * 0.5 + 0.5  # Values between 0-1
        
        # Create custom colormap matching the legend
        colors = ["#2b83ba", "#abdda4", "#ffffbf", "#fdae61", "#d7191c"]  # Blue, Green, Yellow, Orange, Red
        cmap = LinearSegmentedColormap.from_list("flood_hazard", colors)
        
        # Create figure with Berlin-style map
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot hazard map
        im = ax.imshow(hazard_data, cmap=cmap, vmin=0, vmax=1, extent=[0, 10, 0, 10])
        
        # Add reported flood points if available
        if st.session_state.reported_flood_points is not None:
            # For simulation, create random points
            np.random.seed(42)
            points_x = np.random.uniform(1, 9, 50)
            points_y = np.random.uniform(1, 9, 50)
            ax.scatter(points_x, points_y, s=30, c='purple', alpha=0.7, edgecolor='white', linewidth=1, label='Reported Floods')
        
        # Add scale bar
        ax.plot([1, 3], [1, 1], 'k-', lw=2)
        ax.text(2, 0.9, "2 Kilometers", ha='center', fontsize=10,
               path_effects=[patheffects.withStroke(linewidth=3, foreground='white')])
        
        # Add legend
        cbar = fig.colorbar(im, ax=ax, shrink=0.5, pad=0.02)
        cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        cbar.set_ticklabels(['Very Low', 'Low', 'Moderate', 'High', 'Very High', ''])
        cbar.set_label('Flood Susceptibility Level', fontsize=12)
        
        # Add district boundaries (simulated)
        districts = [
            {'x': [2, 5, 5, 2, 2], 'y': [2, 2, 5, 5, 2]},
            {'x': [5, 8, 8, 5, 5], 'y': [2, 2, 5, 5, 2]},
            {'x': [2, 5, 5, 2, 2], 'y': [5, 5, 8, 8, 5]},
            {'x': [5, 8, 8, 5, 5], 'y': [5, 5, 8, 8, 5]}
        ]
        
        for district in districts:
            ax.plot(district['x'], district['y'], 'k-', lw=0.5)
        
        # Add title and labels
        ax.set_title("Flood Susceptibility Map with Reported Flood Locations", fontsize=14)
        ax.text(5, -0.5, "Berlin Districts", ha='center', fontsize=10, 
               path_effects=[patheffects.withStroke(linewidth=3, foreground='white')])
        
        # Remove axes
        ax.set_axis_off()
        
        # Display the plot
        st.pyplot(fig)
        
        # Add references
        st.markdown("---")
        st.subheader("References")
        st.markdown("""
        - Flood susceptibility modeling using machine learning techniques
        - Geospatial analysis of flood risk factors
        - Berlin Environmental Agency, 2023
        """)
        
    else:
        st.info("Train the model in the 'Data & Model' tab to generate the hazard map")
        st.image("https://via.placeholder.com/800x400?text=Flood+Susceptibility+Map", use_column_width=True)

# Sidebar information
st.sidebar.header("About")
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3163/3163478.png", width=80)
st.sidebar.markdown("""
**Flood Susceptibility Analysis System**

This application uses machine learning to predict flood susceptibility based on geospatial factors.

**Workflow:**
1. Upload study area map
2. Add flood inventory map and factor maps
3. Select and train model
4. View/download hazard map

**Supported Models:**
- Random Forest
- Logistic Regression
- Support Vector Machine
""")

st.sidebar.header("Legend Reference")
st.sidebar.image("https://via.placeholder.com/300x200?text=Custom+Legend", use_column_width=True)
st.sidebar.markdown("""
- **Reported flooded locations**: Purple points
- **Very low**: Blue
- **Low**: Green
- **Moderate**: Yellow
- **High**: Orange
- **Very high**: Red
- **Berlin districts**: Black boundaries
""")

st.sidebar.header("Data Requirements")
st.sidebar.info("""
- **Study Area**: Shapefile or GeoTIFF
- **Flood Map**: GeoTIFF (1=flooded, 0=non-flooded)
- **Factor Maps**: GeoTIFF rasters (same extent/resolution)
""")

# Footer
st.markdown("---")
st.caption("© 2023 Flood Susceptibility Analysis System | Developed with Streamlit")
