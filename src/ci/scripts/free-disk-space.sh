#!/bin/bash
set -euo pipefail

# Free disk space on Linux GitHub action runners
# Script inspired by https://github.com/jlumbroso/free-disk-space

# When updating to a new ubuntu version:
# - Check that there are no docker images preinstalled with `docker image ls`
# - Check that there are no big packages preinstalled that we aren't using

# print a line of the specified character
printSeparationLine() {
    for ((i = 0; i < 80; i++)); do
        printf "%s" "$1"
    done
    printf "\n"
}

# compute available space
# REF: https://unix.stackexchange.com/a/42049/60849
# REF: https://stackoverflow.com/a/450821/408734
getAvailableSpace() {
    df -a | awk 'NR > 1 {avail+=$4} END {print avail}'
}

# make Kb human readable (assume the input is Kb)
# REF: https://unix.stackexchange.com/a/44087/60849
formatByteCount() {
    numfmt --to=iec-i --suffix=B --padding=7 "$1"'000'
}

# macro to output saved space
printSavedSpace() {
    # Disk space before the operation
    local before=${1}
    local title=${2:-}

    local after
    after=$(getAvailableSpace)
    local saved=$((after - before))

    echo ""
    printSeparationLine "*"
    if [ -n "${title}" ]; then
        echo "=> ${title}: Saved $(formatByteCount "$saved")"
    else
        echo "=> Saved $(formatByteCount "$saved")"
    fi
    printSeparationLine "*"
    echo ""
}

# macro to print output of df with caption
printDF() {
    local caption=${1}

    printSeparationLine "="
    echo "${caption}"
    echo ""
    echo "$ df -h"
    echo ""
    df -h
    printSeparationLine "="
}

removeDir() {
    dir=${1}

    local before
    if [ ! -d "$dir" ]; then
        echo "::warning::Directory $dir does not exist, skipping."
    else
        before=$(getAvailableSpace)
        sudo rm -rf "$dir"
        printSavedSpace "$before" "Removed $dir"
    fi
}

removeUnusedDirectories() {
    local dirs_to_remove=(
        "/usr/local/lib/android"
        # Haskell runtime
        "/usr/local/.ghcup"
        # Azure
        "/opt/az"
        "/etc/mysql"
        "/usr/share/php"
        "/etc/php/"
    )

    for dir in "${dirs_to_remove[@]}"; do
        removeDir "$dir"
    done
}

execAndMeasureSpaceChange() {
    local operation=${1} # Function to execute
    local title=${2}

    local before
    before=$(getAvailableSpace)
    $operation

    printSavedSpace "$before" "$title"
}

# Remove large packages
# REF: https://github.com/apache/flink/blob/master/tools/azure-pipelines/free_disk_space.sh
cleanPackages() {
    sudo apt-get purge -y --fix-missing \
        '^aspnetcore-.*'        \
        '^dotnet-.*'            \
        '^java-*'               \
        '^libllvm.*'            \
        '^llvm-.*'              \
        '^mysql-.*'             \
        '^vim.*'                \
        'azure-cli'             \
        'firefox'               \
        'gcc'                   \
        'gcc-12'                \
        'gcc-13'                \
        'google-chrome-stable'  \
        'google-cloud-cli'      \
        'groff-base'            \
        'kubectl'               \
        'libgl1-mesa-dri'       \
        'microsoft-edge-stable' \
        'php.*'                 \
        'powershell'            \
        'snapd'

    sudo apt-get autoremove -y || echo "::warning::The command [sudo apt-get autoremove -y] failed"
    sudo apt-get clean || echo "::warning::The command [sudo apt-get clean] failed failed"
}

# Remove Swap storage
cleanSwap() {
    sudo swapoff -a || true
    sudo rm -rf /mnt/swapfile || true
    free -h
}

# Display initial disk space stats

AVAILABLE_INITIAL=$(getAvailableSpace)

printDF "BEFORE CLEAN-UP:"
echo ""

execAndMeasureSpaceChange cleanPackages "Unused packages"
execAndMeasureSpaceChange cleanSwap "Swap storage"

removeUnusedDirectories

# Output saved space statistic
echo ""
printDF "AFTER CLEAN-UP:"

echo ""
echo ""

printSavedSpace "$AVAILABLE_INITIAL" "Total saved"
