#!/bin/bash

# Initialize Conda for Bash shell
eval "$(conda shell.bash hook)"

# Activate the rlenv Conda environment
conda activate rlenv

# Exit if conda activation fails
if [ $? -ne 0 ]; then
    echo "Failed to activate conda environment."
    exit 1
fi

# Change to the required directory
cd examples/hh

# Exit if directory change fails
if [ $? -ne 0 ]; then
    echo "Failed to change directory to examples/hh."
    exit 1
fi

# Run the sac_hh.py script with the accelerate launch command
CONFIG_NAME=125M accelerate launch --num_processes 1 --config_file ../../configs/accelerate/zero2-bf16.yaml sac_hh.py

# Exit if script execution fails
if [ $? -ne 0 ]; then
    echo "Failed to execute sac_hh.py with accelerate."
    exit 1
fi
