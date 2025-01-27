#!/bin/bash
set -euo pipefail

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
    # Prevent npm from being removed.
    # This command unfortunately also prevents npm from being updated,
    # but it should be fine because we don't run `apt upgrade`.
    sudo apt-mark hold npm

    sudo apt-get -qq purge -y --autoremove --fix-missing \
        '^aspnetcore-.*'          \
        '^dotnet-.*'              \
        '^java-*'                 \
        '^libllvm.*'              \
        '^llvm.*'                 \
        '^mongodb-.*'             \
        '^mysql-.*'               \
        '^r-base.*'               \
        '^vim.*'                  \
        'azure-cli'               \
        'cpp-11'                  \
        'firefox'                 \
        'gcc-10'                  \
        'gcc-11'                  \
        'gcc-12'                  \
        'gcc-9'                   \
        'gcc'                     \
        'google-chrome-stable'    \
        'google-cloud-cli'        \
        'google-cloud-sdk'        \
        'groff-base'              \
        'groff'                   \
        'kubectl'                 \
        'libgl1-mesa-dri'         \
        'libicu-dev'              \
        'mercurial-common'        \
        'microsoft-edge-stable'   \
        'mono-devel'              \
        'mono-llvm-tools'         \
        'php.*'                   \
        'podman'                  \
        'powershell'              \
        'python-babel-localedata' \
        'python3-breezy'          \
        'skopeo'                  \
        'snapd'                   \
        'tmux'

    sudo apt-get autoremove -y || echo "::warning::The command [sudo apt-get autoremove -y] failed"
    sudo apt-get clean || echo "::warning::The command [sudo apt-get clean] failed failed"

    echo "=> Installed packages sorted by size:"
    # sort always fails because `head` stops reading stdin
    dpkg-query -W --showformat='${Installed-Size} ${Package}\n' | \
      sort -nr 2>/dev/null | head -200 || true
}

# Remove Docker images
cleanDocker() {
    echo "=> Removing the following docker images:"
    sudo docker image ls
    echo "=> Removing docker images..."
    sudo docker image prune --all --force || true
}

removeAllSnaps() {
    # This won't remove the snaps `core` and `snapd`
    sudo snap remove $(snap list | awk '!/^Name|^core|^snapd/ {print $1}')
}

removeUnusedDirectories() {
    local dirs_to_remove=(
        "/usr/lib/heroku/"
        "/usr/local/lib/android"
        "/usr/local/share/chromium"
        "/usr/local/share/powershell"
        "/usr/share/az_"*
        "/usr/local/share/cmake-"*
        "/usr/share/dotnet"
        "/usr/share/icons/"
        "/usr/share/miniconda/"
        "/usr/share/swift"

        # Environemnt variable set by GitHub Actions
        "$AGENT_TOOLSDIRECTORY"

        # Haskell runtime
        "/opt/ghc"
        "/usr/local/.ghcup"
    )
    local before

    for dir in "${dirs_to_remove[@]}"; do
        before=$(getAvailableSpace)
        sudo rm -rf "$dir" || true
        printSavedSpace "$before" "Removed $dir"
    done

    echo "=> largest directories:"
    # sort always fails because `head` stops reading stdin
    sudo du --max-depth=7 /* -h | sort -nr 2>/dev/null  | head -1000 || true
}

# Display initial disk space stats

AVAILABLE_INITIAL=$(getAvailableSpace)

printDF "BEFORE CLEAN-UP:"
echo ""

execAndMeasureSpaceChange removeAllSnaps "Snaps"
execAndMeasureSpaceChange cleanPackages "Unused packages"
execAndMeasureSpaceChange cleanDocker "Docker images"

removeUnusedDirectories

# Output saved space statistic
echo ""
printDF "AFTER CLEAN-UP:"

echo ""
echo ""

printSavedSpace "$AVAILABLE_INITIAL" "Total saved"
