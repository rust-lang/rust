#!/bin/bash
# Free disk space on Linux GitHub action runners

# # ======
# MACROS
# ======

# macro to print a line of equals
# # (silly but works)
printSeparationLine() {
    str=${1:=}
    num=${2:-80}
    counter=1
    output=""
    while [ $counter -le "$num" ]
    do
        output="${output}${str}"
        counter=$((counter+1))
    done
    echo "${output}"
}

# macro to compute available space
# REF: https://unix.stackexchange.com/a/42049/60849
# REF: https://stackoverflow.com/a/450821/408734
getAvailableSpace() { echo $(df -a $1 | awk 'NR > 1 {avail+=$4} END {print avail}'); }
# macro to make Kb human readable (assume the input is Kb)
# REF: https://unix.stackexchange.com/a/44087/60849
formatByteCount() { echo $(numfmt --to=iec-i --suffix=B --padding=7 $1'000'); }

# macro to output saved space
printSavedSpace() {
    saved=${1}
    title=${2:-}

    echo ""
    printSeparationLine '*' 80
    if [ -n "${title}" ]; then
        echo "=> ${title}: Saved $(formatByteCount "$saved")"
    else
        echo "=> Saved $(formatByteCount "$saved")"
    fi
    printSeparationLine '*' 80
    echo ""
}

# macro to print output of dh with caption
printDH() {
    caption=${1:-}

    printSeparationLine '=' 80
    echo "${caption}"
    echo ""
    echo "$ dh -h /"
    echo ""
    df -h /
    echo "$ dh -a /"
    echo ""
    df -a /
    echo "$ dh -a"
    echo ""
    df -a
    printSeparationLine '=' 80
}

# ======
# SCRIPT
# # ======

# Display initial disk space stats

AVAILABLE_INITIAL=$(getAvailableSpace)
AVAILABLE_ROOT_INITIAL=$(getAvailableSpace '/')

printDH "BEFORE CLEAN-UP:"
echo ""

BEFORE=$(getAvailableSpace)

sudo rm -rf "$AGENT_TOOLSDIRECTORY" || true

AFTER=$(getAvailableSpace)
SAVED=$((AFTER-BEFORE))
printSavedSpace $SAVED "Agent tools"

BEFORE=$(getAvailableSpace)

sudo rm -rf /usr/local/share/powershell || true

AFTER=$(getAvailableSpace)
SAVED=$((AFTER-BEFORE))
printSavedSpace $SAVED "Powershell"

BEFORE=$(getAvailableSpace)

sudo rm -rf /usr/local/share/chromium || true

AFTER=$(getAvailableSpace)
SAVED=$((AFTER-BEFORE))
printSavedSpace $SAVED "Chromium"

BEFORE=$(getAvailableSpace)

sudo rm -rf /usr/local/lib/node_modules || true

AFTER=$(getAvailableSpace)
SAVED=$((AFTER-BEFORE))
printSavedSpace $SAVED "Node modules"

BEFORE=$(getAvailableSpace)

sudo rm -rf /usr/share/swift || true

AFTER=$(getAvailableSpace)
SAVED=$((AFTER-BEFORE))
printSavedSpace $SAVED "Swift"

# Remove Android library
BEFORE=$(getAvailableSpace)

sudo rm -rf /usr/local/lib/android || true

AFTER=$(getAvailableSpace)
SAVED=$((AFTER-BEFORE))
printSavedSpace $SAVED "Android library"

# Remove .NET runtime

BEFORE=$(getAvailableSpace)

# https://github.community/t/bigger-github-hosted-runners-disk-space/17267/11
sudo rm -rf /usr/share/dotnet || true

AFTER=$(getAvailableSpace)
SAVED=$((AFTER-BEFORE))
printSavedSpace $SAVED ".NET runtime"

# Remove Haskell runtime
BEFORE=$(getAvailableSpace)

sudo rm -rf /opt/ghc || true
sudo rm -rf /usr/local/.ghcup || true

AFTER=$(getAvailableSpace)
SAVED=$((AFTER-BEFORE))
printSavedSpace $SAVED "Haskell runtime"

# Remove large packages
# REF: https://github.com/apache/flink/blob/master/tools/azure-pipelines/free_disk_space.sh

BEFORE=$(getAvailableSpace)

sudo apt-get remove -y '^aspnetcore-.*' || echo "::warning::The command [sudo apt-get remove -y '^aspnetcore-.*'] failed to complete successfully. Proceeding..."
sudo apt-get remove -y '^dotnet-.*' --fix-missing || echo "::warning::The command [sudo apt-get remove -y '^dotnet-.*' --fix-missing] failed to complete successfully. Proceeding..."
sudo apt-get remove -y '^llvm-.*' --fix-missing || echo "::warning::The command [sudo apt-get remove -y '^llvm-.*' --fix-missing] failed to complete successfully. Proceeding..."
sudo apt-get remove -y 'php.*' --fix-missing || echo "::warning::The command [sudo apt-get remove -y 'php.*' --fix-missing] failed to complete successfully. Proceeding..."
sudo apt-get remove -y '^mongodb-.*' --fix-missing || echo "::warning::The command [sudo apt-get remove -y '^mongodb-.*' --fix-missing] failed to complete successfully. Proceeding..."
sudo apt-get remove -y '^mysql-.*' --fix-missing || echo "::warning::The command [sudo apt-get remove -y '^mysql-.*' --fix-missing] failed to complete successfully. Proceeding..."
sudo apt-get remove -y azure-cli google-chrome-stable firefox powershell mono-devel libgl1-mesa-dri --fix-missing || echo "::warning::The command [sudo apt-get remove -y azure-cli google-chrome-stable firefox powershell mono-devel libgl1-mesa-dri --fix-missing] failed to complete successfully. Proceeding..."
sudo apt-get remove -y google-cloud-sdk --fix-missing || echo "::debug::The command [sudo apt-get remove -y google-cloud-sdk --fix-missing] failed to complete successfully. Proceeding..."
sudo apt-get remove -y google-cloud-cli --fix-missing || echo "::debug::The command [sudo apt-get remove -y google-cloud-cli --fix-missing] failed to complete successfully. Proceeding..."
sudo apt-get remove -y microsoft-edge-stable --fix-missing || echo "::debug::The command [sudo apt-get remove -y microsoft-edge-stable --fix-missing] failed to complete successfully. Proceeding..."
sudo apt-get remove -y snapd --fix-missing || echo "::debug::The command [sudo apt-get remove -y snapd --fix-missing] failed to complete successfully. Proceeding..."
sudo apt-get autoremove -y || echo "::warning::The command [sudo apt-get autoremove -y] failed to complete successfully. Proceeding..."
sudo apt-get clean || echo "::warning::The command [sudo apt-get clean] failed to complete successfully. Proceeding..."

AFTER=$(getAvailableSpace)
SAVED=$((AFTER-BEFORE))
printSavedSpace $SAVED "Large misc. packages"

# Remove Docker images

BEFORE=$(getAvailableSpace)

sudo docker image prune --all --force || true

AFTER=$(getAvailableSpace)
SAVED=$((AFTER-BEFORE))
printSavedSpace $SAVED "Docker images"

# Remove tool cache
# REF: https://github.com/actions/virtual-environments/issues/2875#issuecomment-1163392159

BEFORE=$(getAvailableSpace)

sudo rm -rf "$AGENT_TOOLSDIRECTORY" || true

AFTER=$(getAvailableSpace)
SAVED=$((AFTER-BEFORE))
printSavedSpace $SAVED "Tool cache"

# Remove Swap storage

BEFORE=$(getAvailableSpace)

sudo swapoff -a || true
sudo rm -rf /mnt/swapfile || true
free -h

AFTER=$(getAvailableSpace)
SAVED=$((AFTER-BEFORE))
printSavedSpace $SAVED "Swap storage"

# Output saved space statistic

AVAILABLE_END=$(getAvailableSpace)
AVAILABLE_ROOT_END=$(getAvailableSpace '/')

echo ""
printDH "AFTER CLEAN-UP:"

echo ""
echo ""

echo "/dev/root:"
printSavedSpace $((AVAILABLE_ROOT_END - AVAILABLE_ROOT_INITIAL))
echo "overall:"
printSavedSpace $((AVAILABLE_END - AVAILABLE_INITIAL))
