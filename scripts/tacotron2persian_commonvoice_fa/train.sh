#! /bin/bash

# get path to config file
# full_path=$0
# CONFIG_PATH=$(dirname "$full_path")
# echo $CONFIG_PATH

# python -m tac2persian.train --config_path="$CONFIG_PATH"


# get path to config file
full_path=$0
CONFIG_PATH=$(dirname "$full_path")
echo "Config path: $CONFIG_PATH"

# default checkpoint path
CHECKPOINT_PATH="outputs/commonvoice_fa/checkpoints/checkpoint_1K_v0.pt"
python3  -m tac2persian.train --config_path="$CONFIG_PATH" --checkpoint_path="$CHECKPOINT_PATH"

