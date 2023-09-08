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

# Run the sac_hh.py script with the accelerate launch command
nohup ./runsac.sh &> sac_log.out &

# Exit if script execution fails
if [ $? -ne 0 ]; then
    echo "Failed to execute nohup runsac.sh"
    exit 1
fi

tail -f sac_log.out

# Exit if tail fails
if [ $? -ne 0 ]; then
    echo "Failed to execute tail -f sac_log.out"
    exit 1
fi