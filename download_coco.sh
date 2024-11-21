#!/bin/bash

# Directory to store the COCO dataset
cd data
DATASET_DIR="coco_dataset"

# URLs for the dataset (modify if needed)
BASE_URL="http://images.cocodataset.org/zips"
TRAIN_URL="${BASE_URL}/train2017.zip"
VAL_URL="${BASE_URL}/val2017.zip"
ANNOTATIONS_URL="http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

# Create the dataset directory if it doesn't exist
mkdir -p $DATASET_DIR

# Function to download and extract a file
download_and_extract() {
    local url=$1
    local dest_dir=$2

    # Extract file name from URL
    local filename=$(basename "$url")
    local filepath="$dest_dir/$filename"

    # Download the file
    echo "Downloading $filename..."
    curl -L -o "$filepath" "$url"

    # Extract the file
    echo "Extracting $filename..."
    unzip -q "$filepath" -d "$dest_dir"

    # Remove the zip file after extraction
    rm "$filepath"
}

# Download and extract the datasets
download_and_extract $TRAIN_URL $DATASET_DIR
download_and_extract $VAL_URL $DATASET_DIR
download_and_extract $ANNOTATIONS_URL $DATASET_DIR

echo "COCO dataset downloaded and extracted to $DATASET_DIR."
cd ..