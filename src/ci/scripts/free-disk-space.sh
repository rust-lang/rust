#!/bin/bash

# Free disk space on Linux GitHub action runners
# Script inspired by https://github.com/jlumbroso/free-disk-space

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
getAvailableSpace() { echo $(df -a | awk 'NR > 1 {avail+=$4} END {print avail}'); }

# make Kb human readable (assume the input is Kb)
# REF: https://unix.stackexchange.com/a/44087/60849
formatByteCount() { echo $(numfmt --to=iec-i --suffix=B --padding=7 $1'000'); }

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
    before=$(getAvailableSpace)

    sudo rm -rf "$dir" || true

    printSavedSpace "$before" "$dir"
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
    sudo apt-get -qq remove -y --fix-missing \
        '^aspnetcore-.*'       \
        '^dotnet-.*'           \
        '^llvm-.*'             \
        'php.*'                \
        '^mongodb-.*'          \
        '^mysql-.*'            \
        'azure-cli'            \
        'google-chrome-stable' \
        'firefox'              \
        'powershell'           \
        'mono-devel'           \
        'libgl1-mesa-dri'      \
        'google-cloud-sdk'     \
        'google-cloud-cli'

    sudo apt-get autoremove -y || echo "::warning::The command [sudo apt-get autoremove -y] failed"
    sudo apt-get clean || echo "::warning::The command [sudo apt-get clean] failed failed"
}

# Remove Docker images
cleanDocker() {
    echo "Removing the following docker images:"
    sudo docker image ls
    echo "Removing docker images..."
    sudo docker image prune --all --force || true
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

removeDir /usr/local/lib/android
removeDir /usr/share/dotnet

# Haskell runtime
removeDir /opt/ghc
removeDir /usr/local/.ghcup

execAndMeasureSpaceChange cleanPackages "Large misc. packages"
execAndMeasureSpaceChange cleanDocker "Docker images"
execAndMeasureSpaceChange cleanSwap "Swap storage"

# Output saved space statistic
echo ""
printDF "AFTER CLEAN-UP:"

echo ""
echo ""

printSavedSpace "$AVAILABLE_INITIAL" "Total saved"
