#!/bin/bash
set -euo pipefail

# Free disk space on Linux GitHub action runners
# Script inspired by https://github.com/jlumbroso/free-disk-space

# When updating to a new ubuntu version (e.g. from ubuntu-24.04):
# - Check that there are no docker images preinstalled with `docker image ls`
# - Check that there are no big packages preinstalled that we aren't using
# - Check that all directores we are removing are still present (look at the warnings)

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

removeRecursive() {
    element=${1}

    local before
    if [ ! -e "$element" ]; then
        echo "::warning::Directory or file $element does not exist, skipping."
    else
        before=$(getAvailableSpace)
        sudo rm -rf "$element"
        printSavedSpace "$before" "Removed $element"
    fi
}

removeUnusedDirsAndFiles() {
    local to_remove=(
        "/etc/mysql"
        "/usr/local/aws-sam-cli"
        "/usr/local/doc/cmake"
        "/usr/local/julia"*
        "/usr/local/lib/android"
        "/usr/local/share/chromedriver-"*
        "/usr/local/share/chromium"
        "/usr/local/share/cmake-"*
        "/usr/local/share/edge_driver"
        "/usr/local/share/gecko_driver"
        "/usr/local/share/icons"
        "/usr/local/share/vim"
        "/usr/local/share/emacs"
        "/usr/local/share/powershell"
        "/usr/local/share/vcpkg"
        "/usr/share/apache-maven-"*
        "/usr/share/gradle-"*
        "/usr/share/java"
        "/usr/share/kotlinc"
        "/usr/share/miniconda"
        "/usr/share/php"
        "/usr/share/ri"
        "/usr/share/swift"

        # binaries
        "/usr/local/bin/azcopy"
        "/usr/local/bin/bicep"
        "/usr/local/bin/ccmake"
        "/usr/local/bin/cmake-"*
        "/usr/local/bin/cmake"
        "/usr/local/bin/cpack"
        "/usr/local/bin/ctest"
        "/usr/local/bin/helm"
        "/usr/local/bin/kind"
        "/usr/local/bin/kustomize"
        "/usr/local/bin/minikube"
        "/usr/local/bin/packer"
        "/usr/local/bin/phpunit"
        "/usr/local/bin/pulumi-"*
        "/usr/local/bin/pulumi"
        "/usr/local/bin/stack"

        # Haskell runtime
        "/usr/local/.ghcup"

        # Azure
        "/opt/az"
        "/usr/share/az_"*

        # Environemnt variable set by GitHub Actions
        "$AGENT_TOOLSDIRECTORY"
    )

    for element in "${to_remove[@]}"; do
        removeRecursive "$element"
    done
}

removeNodeModules() {
    sudo npm uninstall -g \
        "@bazel/bazelisk" \
        "bazel"           \
        "grunt"           \
        "gulp"            \
        "lerna"           \
        "n"               \
        "newman"          \
        "parcel"          \
        "typescript"      \
        "webpack-cli"     \
        "webpack"         \
        "yarn"
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
    sudo apt-get purge -y --autoremove --fix-missing \
        '.*-icon-theme$'         \
        '^aspnetcore-.*'        \
        '^dotnet-.*'            \
        '^java-*'               \
        '^libllvm.*'            \
        '^llvm-.*'              \
        '^mercurial.*'          \
        '^mysql-.*'             \
        '^vim.*'                \
        '^fonts-.*'             \
        'azure-cli'             \
        'buildah'               \
        'cpp-13'                \
        'firefox'               \
        'gcc-12'                \
        'gcc-13'                \
        'gcc-14'                \
        'gcc'                   \
        'g++-14'                \
        'gfortran-14'           \
        'google-chrome-stable'  \
        'google-cloud-cli'      \
        'groff-base'            \
        'kubectl'               \
        'libgl1-mesa-dri'       \
        'microsoft-edge-stable' \
        'php.*'                 \
        'podman'                \
        'powershell'            \
        'skopeo'                \
        'snapd'                 \
        'tmux'

    echo "=> apt-get autoremove"
    sudo apt-get autoremove -y || echo "::warning::The command [sudo apt-get autoremove -y] failed"
    echo "=> apt-get clean"
    sudo apt-get clean || echo "::warning::The command [sudo apt-get clean] failed failed"
}

# Remove Swap storage
cleanSwap() {
    sudo swapoff -a || true
    sudo rm -rf /mnt/swapfile || true
    free -h
}

removePythonPackages() {
    sudo pipx uninstall ansible-core
}

# Display initial disk space stats

AVAILABLE_INITIAL=$(getAvailableSpace)

printDF "BEFORE CLEAN-UP:"
echo ""

execAndMeasureSpaceChange cleanPackages "Unused packages"
execAndMeasureSpaceChange cleanSwap "Swap storage"
execAndMeasureSpaceChange removeNodeModules "Node modules"
execAndMeasureSpaceChange removePythonPackages "Python Packages"

removeUnusedDirsAndFiles

# Output saved space statistic
echo ""
printDF "AFTER CLEAN-UP:"

echo ""
echo ""

printSavedSpace "$AVAILABLE_INITIAL" "Total saved"
