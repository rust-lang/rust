#!/bin/bash
set -euo pipefail
# ignore-tidy-file-linelength

# Free disk space on macOS GitHub action runners
# Achieved by removing all iOS simulator instance files and runtimes
# See https://developer.apple.com/documentation/xcode/xcode-command-line-tool-reference#Manage-the-Simulator

space_before=$(df -k "$GITHUB_WORKSPACE" | awk 'NR == 2 {print $4}')

xcrun simctl shutdown all
xcrun simctl delete all

# This subcommand runs asynchronously
xcrun simctl runtime delete all

# Wait a max of 60 seconds to ensure the clean-up is done before any subsequential build steps
for ((attempt = 0; attempt < 30; attempt++)); do
    remaining=$(xcrun simctl runtime list -j | jq 'length')
    echo "Remaining runtime iOS simulator images : $remaining"

    if [[ "$remaining" == 0 ]]; then
        break
    fi
    sleep 2
done

if [[ "$remaining" != 0 ]]; then
    echo "::warning::Runtime deletion did not finish within the polling period."
fi

# Report net increase in available disk space
space_after=$(df -k "$GITHUB_WORKSPACE" | awk 'NR == 2 {print $4}')
extra_space=$(awk -v b="$space_before" -v a="$space_after" 'BEGIN { printf "%.2f", (a - b) / 1048576 }')
echo "Increase in available disk space: $extra_space GiB"
