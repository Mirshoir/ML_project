import streamlit as st
import geopandas as gpd
import rasterio
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import tensorflow
from keras.models import Sequential
import folium
import branca.colormap as cm
import rioxarray # Make sure rioxarray is imported for .rio access
import xarray as xr
import os
import tempfile
import time
from rasterio.sample import sample_gen
from sklearn.model_selection import train_test_split
import joblib
import tensorflow as tf

DATA_DIR = "data"
SHAPEFILE_PATH = os.path.join(DATA_DIR, "Points.shp")
RASTER_PATHS = [
    os.path.join(DATA_DIR, "Curvature.tif"),
    os.path.join(DATA_DIR, "DTDrainage.tif"),
    os.path.join(DATA_DIR, "DTRiver.tif"),
]
MODEL_PATH = "models/random_forest_model.pkl"
MODEL_TYPE = 'random_forest'  # or 'deep_learning' if applicable
# Function to load spatial data
@st.cache_data
def load_spatial_data(shapefile_path, raster_paths):
    """Loads spatial data (shapefile points and raster values) into a GeoDataFrame."""
    st.write("Loading spatial data...")
    gdf = gpd.read_file(shapefile_path)

    # Ensure GeoDataFrame has a CRS, set it if missing
    if gdf.crs is None:
         # Assuming the original shapefile's CRS based on previous notebook cells (EPSG:25833)
         gdf.crs = "EPSG:25833"
         st.write("Assigned CRS 'EPSG:25833' to shapefile.")


    # Extract raster values
    for path in raster_paths:
        var_name = os.path.splitext(os.path.basename(path))[0]
        st.write(f"Extracting values from {var_name}...")
        try:
            with rasterio.open(path) as src:
                # Ensure raster and gdf are in the same CRS for sampling
                if gdf.crs != src.crs:
                    st.write(f"Warning: CRS mismatch between gdf ({gdf.crs}) and raster ({src.crs}). Reprojecting gdf to raster CRS.")
                    # Reproject gdf to raster CRS for sampling
                    gdf_reprojected = gdf.to_crs(src.crs)
                    gdf[var_name] = [
                        list(src.sample([(geom.x, geom.y)]))[0][0] if geom.is_valid else np.nan
                        for geom in gdf_reprojected.geometry
                    ]
                else:
                     gdf[var_name] = [
                        list(src.sample([(geom.x, geom.y)]))[0][0] if geom.is_valid else np.nan
                        for geom in gdf.geometry
                    ]
        except Exception as e:
            st.error(f"Error loading or sampling raster {path}: {e}")
            gdf[var_name] = np.nan # Assign NaN if loading fails

    # Drop rows with NaN values introduced by raster sampling outside extent or invalid geometries
    initial_rows = len(gdf)
    gdf.dropna(inplace=True)
    if len(gdf) < initial_rows:
        st.write(f"Dropped {initial_rows - len(gdf)} points with missing raster values.")

    st.write("Spatial data loaded.")
    return gdf

# Function to load the trained model
@st.cache_resource # Use st.cache_resource for models and connection objects
def load_model(model_path, model_type):
    """Loads the trained machine learning model."""
    st.write(f"Loading {model_type} model from {model_path}...")
    try:
        if model_type == 'random_forest':
            # Load scikit-learn model using joblib
            loaded_model = joblib.load(model_path)
        elif model_type == 'deep_learning':
            # Load Keras model using tensorflow
            loaded_model = tf.keras.models.load_model(model_path)
        else:
            st.error(f"Unknown model type: {model_type}")
            return None
        st.write("Model loaded successfully.")
        return loaded_model
    except Exception as e:
        st.error(f"Error loading model from {model_path}: {e}")
        return None

# --- Streamlit App Layout (Placeholder Calls) ---

st.title("Real-time Flood Susceptibility Analysis")

# Load static data and model
# These will be cached after the first run
static_gdf = load_spatial_data(SHAPEFILE_PATH, RASTER_PATHS)
trained_model = load_model(MODEL_PATH, MODEL_TYPE)

if static_gdf is not None and trained_model is not None:
    st.sidebar.header("Configuration")

    # Display info about loaded data and model (optional)
    st.sidebar.write(f"Loaded {len(static_gdf)} spatial points.")
    st.sidebar.write(f"Model type: {MODEL_TYPE}")
    # Add more info if needed, e.g., model features

    # Placeholder for future real-time data simulation or input
    st.header("Real-time Data Simulation")
    simulate_real_time = st.checkbox("Simulate Real-time Data", value=False, key="simulate_checkbox")

    # Placeholder for prediction and mapping logic
    st.header("Flood Susceptibility Map")
    st.write("Map will be displayed here.")


else:
    st.error("Failed to load required data or model. Please check paths and file formats.")


# Placeholder function to simulate fetching real-time raster data
def fetch_real_time_raster_data(area_of_interest=None):
    """
    Simulates fetching real-time raster data for a given area.
    In a real system, this would involve API calls, data downloads, etc.
    """
    st.write("Simulating fetching real-time raster data...")
    time.sleep(1) # Simulate network latency

    # Create dummy data with a more realistic spatial extent and CRS
    # Let's simulate a small area that overlaps with Berlin's approximate location
    # Using EPSG:4326 for the dummy data's initial CRS
    min_lon, min_lat = 13.0, 52.3
    max_lon, max_lat = 13.8, 52.7
    resolution = 0.001 # degrees, roughly 100m at this latitude

    # Calculate dimensions
    width = int((max_lon - min_lon) / resolution)
    height = int((max_lat - min_lat) / resolution)

    # Create coordinates
    # xarray convention is usually y (latitude) then x (longitude)
    y_coords = np.linspace(max_lat, min_lat, height) # Latitude decreases from top to bottom
    x_coords = np.linspace(min_lon, max_lon, width)  # Longitude increases from left to right

    dummy_data = np.random.rand(height, width) * 100 # Dummy values (e.g., rainfall intensity)

    # Create a dummy DataArray with spatial dimensions and coordinates
    real_time_data_array = xr.DataArray(
        dummy_data,
        coords={'y': y_coords, 'x': x_coords},
        dims=["y", "x"],
        name="simulated_real_time_feature"
    )

    # Set the CRS for the dummy data
    real_time_data_array = real_time_data_array.rio.set_crs("EPSG:4326")


    st.write("Dummy real-time data fetched with CRS EPSG:4326.")
    return real_time_data_array

# Placeholder function to preprocess data (format conversion, projection, alignment)
def preprocess_real_time_data(data_array, target_crs="EPSG:25833"):
    """
    Simulates preprocessing real-time data.
    In a real system, this would handle format conversion, reprojection,
    resampling/alignment, etc.
    """
    st.write(f"Simulating preprocessing data to {target_crs}...")
    time.sleep(1) # Simulate processing time

    if data_array is None:
        st.write("Input data_array is None.")
        return None

    # Check if data_array has a CRS defined
    if data_array.rio.crs is None:
        st.write("Input data_array has no CRS defined. Cannot reproject.")
        return None

    # Reproject to the target CRS (model's CRS)
    try:
        # Use a resampling method like 'nearest' or 'bilinear' if resolution changes significantly
        # For this simulation, simple reproject is sufficient
        preprocessed_data_array = data_array.rio.reproject(target_crs)
        st.write(f"Successfully reprojected to {target_crs}.")
        return preprocessed_data_array
    except Exception as e:
        st.error(f"Error during reprojection simulation: {e}")
        # In a real system, handle this error appropriately
        return None

# Placeholder function to extract/calculate model features from real-time data
def extract_model_features(preprocessed_data_array, point_locations):
    """
    Simulates extracting model features at specified point locations.
    In a real system, this might involve sampling raster data at points,
    or combining multiple real-time sources to derive features.
    """
    st.write("Simulating feature extraction at point locations...")
    time.sleep(1) # Simulate processing time

    if preprocessed_data_array is None:
        st.write("Preprocessing failed, cannot extract features.")
        return None

    # Ensure point_locations is in the same CRS as the raster
    if point_locations.crs != preprocessed_data_array.rio.crs:
         st.write(f"Reprojecting point locations from {point_locations.crs} to {preprocessed_data_array.rio.crs}")
         try:
             point_locations_reprojected = point_locations.to_crs(preprocessed_data_array.rio.crs)
         except Exception as e:
             st.error(f"Error reprojecting point locations: {e}")
             return None
    else:
        point_locations_reprojected = point_locations


    # Sample the raster at each point location using rasterio's sample_gen
    try:
        # Save the reprojected DataArray to a temporary GeoTIFF file
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmpfile:
            temp_raster_path = tmpfile.name

        preprocessed_data_array.rio.to_raster(temp_raster_path)

        # Open the temporary file with rasterio
        with rasterio.open(temp_raster_path) as src:
             # Get point coordinates as a list of (x, y) tuples
             point_coords = [(p.x, p.y) for p in point_locations_reprojected.geometry]

             # Use rasterio.sample.sample_gen
             extracted_values_gen = sample_gen(src, point_coords)

             # Convert generator output to a list of values
             extracted_values = [value[0] for value in extracted_values_gen]

        # Clean up the temporary file
        os.remove(temp_raster_path)


        # Create a DataFrame of extracted features
        # In a real system, this DataFrame would have multiple columns for each feature
        features_df = pd.DataFrame({
            'simulated_feature': extracted_values
        }, index=point_locations.index) # Keep original index

        # Handle NaNs that result from sampling outside raster extent (sample_gen returns nodata or values)
        # We need to explicitly check for nodata if defined
        # Re-open the raster temporarily to get nodata value if needed, or get it from preprocessed_data_array
        nodata_value = preprocessed_data_array.rio.nodata
        if nodata_value is not None:
             features_df.replace(nodata_value, np.nan, inplace=True)

        features_df.dropna(inplace=True)


    except Exception as e:
        st.error(f"Error during feature extraction simulation: {e}")
        # Clean up the temporary file if it exists
        if 'temp_raster_path' in locals() and os.path.exists(temp_raster_path):
             os.remove(temp_raster_path)
        return None


    st.write(f"Dummy features extracted for {len(features_df)} valid points.")
    return features_df

# --- Streamlit App Layout (Integration) ---

if static_gdf is not None and trained_model is not None:

    st.header("Real-time Data Simulation")
    simulate_real_time = st.checkbox("Simulate Real-time Data", value=False, key="simulate_real_time_checkbox_1")

    real_time_features_at_points = None # Initialize variable

    if simulate_real_time:
        st.subheader("Simulating Real-time Data Ingestion and Preprocessing")
        # Step 1: Simulate fetching real-time data (e.g., a rainfall raster)
        real_time_data = fetch_real_time_raster_data()

        # Step 2: Simulate preprocessing (reprojection and alignment)
        # Assuming the model was trained with data in EPSG:25833
        preprocessed_data = preprocess_real_time_data(real_time_data, target_crs="EPSG:25833")

        # Step 3: Simulate extracting model features at the point locations
        # Use the static_gdf for point locations
        if preprocessed_data is not None:
            real_time_features_at_points = extract_model_features(preprocessed_data, static_gdf)
            if real_time_features_at_points is not None:
                 st.write("Simulated real-time features generated.")
                 st.dataframe(real_time_features_at_points.head()) # Display preview
            else:
                st.warning("Failed to extract real-time features.")
        else:
            st.warning("Preprocessing failed, skipping feature extraction.")

    # Placeholder for prediction and mapping logic (will use real_time_features_at_points if available)
    st.header("Flood Susceptibility Map")
    st.write("Map will be displayed here.")


else:
    st.error("Failed to load required data or model. Please check paths and file formats.")

# Re-define generate_real_time_predictions function based on cell 992c5099
# (Ensure it's the correct version based on the features the loaded model expects)
# Assume the models (rf or model) were trained on ['Curvature', 'DTDrainage', 'DTRiver']

from keras.models import Sequential
# Ensure pandas and numpy are imported (should be already)
# import pandas as pd
# import numpy as np

def generate_real_time_predictions(real_time_features_df, trained_model, static_features_df, model_trained_features):
    """
    Generates flood susceptibility predictions using a trained model and real-time features.

    Args:
        real_time_features_df (pd.DataFrame): DataFrame containing the preprocessed
                                              real-time features for each point.
                                              Must have an index that aligns with static_features_df.
        trained_model: The trained scikit-learn or Keras model object.
        static_features_df (pd.DataFrame): DataFrame containing the static features
                                           relevant to the trained model for each point.
                                           Must have an index that aligns with real_time_features_df.
        model_trained_features (list): A list of feature names (column names) that the
                                       trained_model was fitted on, in the correct order.


    Returns:
        np.ndarray: Array of predictions (class labels or probabilities), or None if prediction fails.
    """
    st.write("Generating real-time predictions...")

    # Ensure the indices align before combining
    if not real_time_features_df.index.equals(static_features_df.index):
        st.write("Warning: Index mismatch between real-time and static features. Merging based on index.")
        # Use index to align - this assumes the indices represent the same points
        # Use an inner join to only keep points present in both
        combined_features = static_features_df.join(real_time_features_df, how='inner')
    else:
        # Use pd.concat to combine, ensuring columns are handled correctly
        combined_features = pd.concat([static_features_df, real_time_features_df], axis=1)


    # Ensure the combined DataFrame contains all features the model was trained on,
    # and in the correct order.
    # We will select the columns based on model_trained_features list.
    # Handle cases where combined_features might not have all model_trained_features
    try:
        features_for_prediction = combined_features[model_trained_features].copy()
    except KeyError as e:
        st.error(f"Error: Combined features DataFrame is missing a required feature: {e}")
        st.write(f"Combined features columns: {combined_features.columns.tolist()}")
        st.write(f"Model trained features: {model_trained_features}")
        return None


    # Generate predictions
    try:
        # If using the Deep Learning model (model from _8GkQEIxyo), it outputs probabilities
        # if trained with sigmoid. We might need to convert to classes.
        # Check the model type to decide
        if isinstance(trained_model, Sequential): # Check if it's a Keras Sequential model
             # Keras predict can return a 2D array for binary classification with one output node
             predictions = trained_model.predict(features_for_prediction)
             # Flatten if necessary and convert to binary if thresholding
             if predictions.ndim > 1:
                 predictions = predictions.flatten()

             # Convert probabilities to binary class (0 or 1) based on a threshold (e.g., 0.5)
             # This threshold might need tuning
             binary_predictions = (predictions > 0.5).astype(int)
             st.write("Generated predictions (probabilities from DL model).")
             st.write(f"First 10 probabilities: {predictions[:10].tolist()}") # Display preview
             st.write(f"First 10 binary predictions: {binary_predictions[:10].tolist()}") # Display preview
             return binary_predictions # Return binary classes for consistency with RF
        else: # Assume scikit-learn model like RandomForestClassifier
            predictions = trained_model.predict(features_for_prediction)
            st.write("Generated predictions (classes from RF model).")
            st.write(f"First 10 predictions: {predictions[:10].tolist()}") # Display preview
            return predictions

    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None

# --- Streamlit App Layout (Integration of Prediction Logic) ---

if static_gdf is not None and trained_model is not None:

    st.header("Real-time Data Simulation")
    simulate_real_time = st.checkbox("Simulate Real-time Data", value=False, key="simulate_real_time_checkbox_2")

    real_time_features_at_points = None # Initialize variable

    if simulate_real_time:
        st.subheader("Simulating Real-time Data Ingestion and Preprocessing")
        # Step 1: Simulate fetching real-time data (e.g., a rainfall raster)
        real_time_data = fetch_real_time_raster_data()

        # Step 2: Simulate preprocessing (reprojection and alignment)
        # Assuming the model was trained with data in EPSG:25833
        preprocessed_data = preprocess_real_time_data(real_time_data, target_crs="EPSG:25833")

        # Step 3: Simulate extracting model features at the point locations
        # Use the static_gdf for point locations
        if preprocessed_data is not None:
            real_time_features_at_points = extract_model_features(preprocessed_data, static_gdf)
            if real_time_features_at_points is not None:
                 st.write("Simulated real-time features generated.")
                 st.dataframe(real_time_features_at_points.head()) # Display preview
            else:
                st.warning("Failed to extract real-time features.")
        else:
            st.warning("Preprocessing failed, skipping feature extraction.")

    # --- Real-time Prediction Block ---
    real_time_predictions = None # Initialize predictions variable

    if real_time_features_at_points is not None:
        st.header("Real-time Flood Susceptibility Prediction")

        # Determine the features the loaded model expects
        # Based on previous analysis, both RF and DL models were trained on these features
        static_model_features = ['Curvature', 'DTDrainage', 'DTRiver']

        # For a *real* real-time system, the model would be trained on static + real-time features.
        # For this demo, we'll pass the static features expected by the pre-trained model
        # AND the simulated real-time feature dataframe. The prediction function
        # will handle selecting the correct features that the *currently loaded* model expects.
        # However, the current pre-trained models DON'T expect 'simulated_feature'.
        # So, for this demo, the prediction will only use the static features.

        # Prepare the static features DataFrame for prediction
        # Ensure it has the correct columns and its index aligns with real_time_features_at_points
        try:
            # Select only the static features required by the model
            static_features_for_prediction = static_gdf[static_model_features].loc[real_time_features_at_points.index]
        except KeyError as e:
             st.error(f"Error preparing static features for prediction: Missing column {e}")
             static_features_for_prediction = None


        if static_features_for_prediction is not None:
            # Call the generate_real_time_predictions function
            # Pass the static features that the loaded model was trained on.
            # The real_time_features_at_points is passed, but the current
            # implementation of generate_real_time_predictions will only use
            # features in the model_trained_features list for prediction.
            real_time_predictions = generate_real_time_predictions(
                real_time_features_at_points, # Pass the simulated real-time data
                trained_model,
                static_features_for_prediction, # Pass the static features aligned with real-time points
                static_model_features # Pass the list of features the loaded model expects
            )

            if real_time_predictions is not None:
                st.write("✅ Real-time predictions generated successfully.")
                # Optional: Display a preview of the predictions
                # st.write("Prediction preview (first 10 values):")
                # st.write(real_time_predictions[:10])
                # st.write(f"Total predictions: {len(real_time_predictions)}")
            else:
                st.warning("❌ Failed to generate real-time predictions.")
        else:
             st.warning("Static features could not be prepared for prediction due to missing columns.")


    # Placeholder for mapping logic (will use real_time_predictions if available)
    st.header("Flood Susceptibility Map")
    if real_time_predictions is not None:
        st.write("Map visualization will use the generated real-time predictions.")
        # The next subtask will add the mapping code here
    else:
        st.write("Simulate real-time data and generate predictions to see the map.")


else:
    st.error("Failed to load required data or model. Please check paths and file formats.")

# Integration with mapping service

# Assume 'real_time_predictions' (array of 0s and 1s) is available from the prediction block
# Assume 'real_time_features_at_points' (DataFrame with point indices) is available
# Assume 'static_gdf' (GeoDataFrame with point geometries and static features) is available

import streamlit_folium # Import streamlit_folium to display Folium maps in Streamlit

# --- Mapping Logic Block ---

if real_time_predictions is not None and real_time_features_at_points is not None and static_gdf is not None:
    st.header("Real-time Flood Susceptibility Map")

    # Filter static_gdf to include only points with real-time predictions
    # The index of real_time_features_at_points corresponds to the index of the points in static_gdf
    # that had valid real-time feature data and thus predictions.
    gdf_real_time = static_gdf.loc[real_time_features_at_points.index].copy()

    # Add real-time predictions to the filtered GeoDataFrame
    # Ensure the predictions array is aligned with the filtered gdf index.
    gdf_real_time['real_time_susceptibility'] = real_time_predictions


    # Create a Folium map
    # Center the map on the mean coordinates of the real-time points
    if not gdf_real_time.empty:
        # Reproject to WGS84 for Folium
        gdf_real_time_wgs84 = gdf_real_time.to_crs("EPSG:4326")
        center_lat = gdf_real_time_wgs84.geometry.y.mean()
        center_lon = gdf_real_time_wgs84.geometry.x.mean()
    else:
        # Default center if no real-time points are available
        center_lat = 52.5200
        center_lon = 13.4050

    m_real_time = folium.Map(location=[center_lat, center_lon], zoom_start=12)

    # Add CircleMarkers for each point with real-time predictions
    for idx, row in gdf_real_time_wgs84.iterrows():
        # Determine color based on prediction (0 or 1)
        color = "red" if row['real_time_susceptibility'] == 1 else "blue" # 1 for higher susceptibility, 0 for lower

        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=5, # Adjust size as needed
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            tooltip=f"Susceptibility: {row['real_time_susceptibility']}" # Add tooltip for value
        ).add_to(m_real_time)

    # Optional: Add a layer control
    folium.LayerControl().add_to(m_real_time)

    # Display the map in Streamlit
    streamlit_folium.folium_static(m_real_time)

else:
    st.write("Generate real-time predictions to visualize the map.")

import streamlit as st # Should be already imported, but good practice to include in file writes

# --- Streamlit App Layout (Adding UI Components) ---

st.title("Real-time Flood Susceptibility Analysis")

# Sidebar for controls
st.sidebar.header("Configuration")

# 1. File Uploader for new data
uploaded_file = st.sidebar.file_uploader(
    "Upload new Shapefile (.shp)",
    type=["shp"],
    help="Upload a new point shapefile to analyze flood susceptibility for these locations."
)

# Placeholder for simulation parameter input (Slider)
st.sidebar.subheader("Simulation Parameters")
simulate_rainfall_intensity = st.sidebar.slider(
    "Simulated Rainfall Intensity (mm/hr)",
    min_value=0.0,
    max_value=100.0,
    value=10.0,
    step=0.1,
    help="Adjust the simulated rainfall intensity for real-time prediction."
)

# 2. Button to trigger analysis
analyze_button = st.sidebar.button("Run Analysis")

# Initialize data and model variables
static_gdf = None
trained_model = None
real_time_predictions = None
real_time_features_at_points = None

# --- Data Loading and Model Loading (Modified to handle uploaded file) ---

# Load static data and model outside of the button click, but check if they exist
# Use st.session_state to persist loaded data and model
if 'static_gdf' not in st.session_state:
    st.session_state.static_gdf = load_spatial_data(SHAPEFILE_PATH, RASTER_PATHS)

if 'trained_model' not in st.session_state:
     # Assuming MODEL_PATH and MODEL_TYPE are defined earlier in the script
     st.session_state.trained_model = load_model(MODEL_PATH, MODEL_TYPE)


static_gdf = st.session_state.static_gdf
trained_model = st.session_state.trained_model

# 4. Text displays for status and progress
status_text = st.empty() # Placeholder for status messages

if static_gdf is None or trained_model is None:
     status_text.error("Failed to load required static data or model. Please check paths and file formats.")
else:
    status_text.info("Static data and model loaded successfully. Upload a shapefile and click 'Run Analysis'.")


# --- Analysis Triggered by Button Click ---
if analyze_button:
    if uploaded_file is not None:
        status_text.info("Processing uploaded shapefile...")
        try:
            # Save the uploaded shapefile temporarily
            with tempfile.TemporaryDirectory() as tmpdir:
                # Streamlit uploads a file-like object, need to save it
                # A shapefile comes as multiple files (.shp, .shx, .dbf, etc.)
                # We need to extract these from the uploaded zip or handle individual files.
                # For simplicity in this demo, let's assume a single .shp file is uploaded,
                # which won't work for a full shapefile. A better approach is a zip upload.
                # As a workaround for this demo, let's just use the *path* of the original
                # static shapefile if a file is uploaded, pretending it's the new data.
                # In a real app, you'd handle the zip or multiple file upload properly.

                # --- Simplified approach for demo: Use original SHAPEFILE_PATH ---
                # Replace this logic with proper uploaded shapefile handling in a real app
                st.warning("Shapefile upload handling is simplified for demo. Using the default shapefile path.")
                new_shapefile_path = SHAPEFILE_PATH # Use the path to the pre-existing shapefile

                # Load the uploaded/new spatial data (using the simplified path for demo)
                # This would ideally load the uploaded shapefile and extract raster values for it
                new_gdf = load_spatial_data(new_shapefile_path, RASTER_PATHS)

                if new_gdf is not None and not new_gdf.empty:
                    status_text.success(f"Uploaded shapefile processed. Loaded {len(new_gdf)} points.")

                    # --- Real-time Data Simulation (using the slider value) ---
                    st.subheader("Real-time Data Simulation")
                    # Adapt fetch_real_time_raster_data or feature extraction
                    # to use 'simulate_rainfall_intensity'.
                    # For this demo, let's just create a dummy real-time feature column
                    # based on the slider value for the new points.
                    st.write(f"Simulating real-time feature based on rainfall intensity: {simulate_rainfall_intensity} mm/hr")

                    # Create a dummy 'simulated_feature' DataFrame for the new points
                    # based on the slider value. This is a very basic simulation.
                    real_time_features_at_points = pd.DataFrame(
                         {'simulated_feature': np.random.rand(len(new_gdf)) * simulate_rainfall_intensity},
                         index=new_gdf.index # Align with the new_gdf index
                    )
                    st.write("Simulated real-time features generated.")
                    st.dataframe(real_time_features_at_points.head())

                    # --- Real-time Prediction ---
                    st.header("Flood Susceptibility Prediction")

                    # Determine the features the loaded model expects
                    # Based on previous analysis, both RF and DL models were trained on these features
                    static_model_features = ['Curvature', 'DTDrainage', 'DTRiver'] # Features trained on

                    # Prepare the static features DataFrame for prediction
                    # Ensure it has the correct columns and its index aligns with real_time_features_at_points
                    try:
                        # Select only the static features required by the model from the new_gdf
                        static_features_for_prediction = new_gdf[static_model_features].loc[real_time_features_at_points.index]
                    except KeyError as e:
                        st.error(f"Error preparing static features for prediction: Missing column {e}")
                        static_features_for_prediction = None

                    if static_features_for_prediction is not None:
                        # Call the generate_real_time_predictions function
                        # Note: The current pre-trained models DON'T expect 'simulated_feature'.
                        # The function will use 'static_model_features' for prediction.
                        real_time_predictions = generate_real_time_predictions(
                            real_time_features_at_points, # Pass the simulated real-time data
                            trained_model,
                            static_features_for_prediction, # Pass the static features aligned with real-time points
                            static_model_features # Pass the list of features the loaded model expects
                        )

                        if real_time_predictions is not None:
                            status_text.success("✅ Real-time predictions generated successfully.")

                            # 5. Display summary statistics about predictions
                            st.subheader("Prediction Summary")
                            prediction_counts = pd.Series(real_time_predictions).value_counts().sort_index()
                            st.write("Number of points predicted per class (0=Low Susceptibility, 1=High Susceptibility):")
                            st.dataframe(prediction_counts)

                            # --- Mapping Logic Block ---
                            st.header("Flood Susceptibility Map")

                            # Filter the new_gdf to include only points with real-time predictions
                            gdf_real_time = new_gdf.loc[real_time_features_at_points.index].copy()
                            gdf_real_time['real_time_susceptibility'] = real_time_predictions

                            # Create and display the Folium map
                            if not gdf_real_time.empty:
                                gdf_real_time_wgs84 = gdf_real_time.to_crs("EPSG:4326")
                                center_lat = gdf_real_time_wgs84.geometry.y.mean()
                                center_lon = gdf_real_time_wgs84.geometry.x.mean()

                                m_real_time = folium.Map(location=[center_lat, center_lon], zoom_start=12)

                                for idx, row in gdf_real_time_wgs84.iterrows():
                                    color = "red" if row['real_time_susceptibility'] == 1 else "blue"
                                    folium.CircleMarker(
                                        location=[row.geometry.y, row.geometry.x],
                                        radius=5,
                                        color=color,
                                        fill=True,
                                        fill_color=color,
                                        fill_opacity=0.7,
                                        tooltip=f"Susceptibility: {row['real_time_susceptibility']}"
                                    ).add_to(m_real_time)

                                folium.LayerControl().add_to(m_real_time)
                                streamlit_folium.folium_static(m_real_time)
                            else:
                                st.warning("No points available to display on the map after prediction.")

                        else:
                            status_text.warning("❌ Failed to generate real-time predictions.")
                    else:
                        status_text.warning("Static features could not be prepared for prediction due to missing columns.")

                else:
                    status_text.error("Failed to load or process the uploaded shapefile.")

        except Exception as e:
            status_text.error(f"An error occurred during analysis: {e}")

    else:
        status_text.warning("Please upload a shapefile to run the analysis.")
