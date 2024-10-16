#! /bin/bash

# Get full path to the config file automatically
full_path=$0
CONFIG_PATH=$(dirname "$full_path")
echo $CONFIG_PATH


DATASET_PATH="PATH TO DATASET"
OUTPUT_PATH="/output_data_atefeh"
NUM_WORKERS=4
META_PATH='PATH OF META CSV'

python -m tac2persian.data_preprocessing.preprocess_commonvoice_fa_df --dataset_path="$DATASET_PATH" \
                                                                   --output_path="$OUTPUT_PATH" \
                                                                   --config_path="$CONFIG_PATH" \
                                                                   --num_workers="$NUM_WORKERS"  \
                                                                   --meta_path="$META_PATH"