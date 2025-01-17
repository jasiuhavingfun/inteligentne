import os
from zipfile import ZipFile

# Path to your dataset
dataset_path = "archive.zip"

# Extract dataset
with ZipFile(dataset_path, 'r') as zip_ref:
    zip_ref.extractall("plant_disease_data")

base_dir = "inteligentne/"  # Adjust based on the dataset folder structure
