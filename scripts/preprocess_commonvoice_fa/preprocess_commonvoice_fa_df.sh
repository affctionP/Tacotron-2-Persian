#! /bin/bash

# Get full path to the config file automatically
full_path=$0
CONFIG_PATH=$(dirname "$full_path")
echo $CONFIG_PATH

# Assign arguments passed when running the script
DATASET_PATH=$1
META_PATH=$2

# Fixed variables
OUTPUT_PATH="/output_data_atefeh"
NUM_WORKERS=4

# Check if the required arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <DATASET_PATH> <META_PATH>"
    exit 1
fi

python -m tac2persian.data_preprocessing.preprocess_commonvoice_fa_df --dataset_path="$DATASET_PATH" \
                                                                   --output_path="$OUTPUT_PATH" \
                                                                   --config_path="$CONFIG_PATH" \
                                                                   --num_workers="$NUM_WORKERS"  \
                                                                   --meta_path="$META_PATH"
