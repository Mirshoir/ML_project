import os
import zipfile
import gdown

# Step 1: Create directory structure
folders = [
    "flood-predictor",
    "flood-predictor/.streamlit",
    "flood-predictor/models",
    "flood-predictor/data"
]

files = {
    "flood-predictor/streamlit_app.py": "# Replace this with your actual Streamlit app code\n",
    "flood-predictor/requirements.txt": """streamlit
geopandas
rasterio
rioxarray
xarray
scikit-learn
tensorflow
pandas
numpy
folium
branca
streamlit-folium
joblib
gdown
""",
    "flood-predictor/README.md": """# 🌊 Real-time Flood Susceptibility Prediction

Deploy this app on Streamlit Cloud with spatial data and ML models.

## Setup

- Place trained model under `models/random_forest_model.pkl`
- Place raster and shapefiles in `data/` folder
""",
    "flood-predictor/.streamlit/config.toml": """[server]
headless = true
enableCORS = false
port = 8501

[theme]
base = "light"
primaryColor = "#4b8bbe"
"""
}

# Create folders
for folder in folders:
    os.makedirs(folder, exist_ok=True)

# Create placeholder files
for path, content in files.items():
    with open(path, "w") as f:
        f.write(content)

# Step 2: Download ZIP data from Google Drive
zip_url = "https://drive.google.com/uc?id=1h1h8DpciFeaL-nQ0PBEzZlVFWFvrdS8-"
output_path = "flood-predictor/data/flood_data.zip"

print("📥 Downloading dataset...")
gdown.download(zip_url, output_path, quiet=False)

# Step 3: Extract the ZIP
print("📂 Extracting files...")
with zipfile.ZipFile(output_path, 'r') as zip_ref:
    zip_ref.extractall("flood-predictor/data")

os.remove(output_path)
print("✅ Done! Project is ready in `flood-predictor/`")
