#!/bin/bash
# Runs bootstrap (in dry-run mode) with each default config profile to ensure they are not broken.

set -euo pipefail

config_dir="../src/bootstrap/defaults"

# Loop through each configuration file in the directory
for config_file in "$config_dir"/*.toml;
do
    python3 ../x.py check --config $config_file --dry-run
done
