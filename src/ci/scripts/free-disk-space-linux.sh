#!/bin/bash
set -euo pipefail

# Free disk space on Linux GitHub action runners
# Script inspired by https://github.com/jlumbroso/free-disk-space

isX86() {
    local arch
    arch=$(uname -m)
    if [ "$arch" = "x86_64" ]; then
        return 0
    else
        return 1
    fi
}

# Check if we're on a GitHub hosted runner.
# In aws codebuild, the variable RUNNER_ENVIRONMENT is "self-hosted".
isGitHubRunner() {
    # `:-` means "use the value of RUNNER_ENVIRONMENT if it exists, otherwise use an empty string".
    if [[ "${RUNNER_ENVIRONMENT:-}" == "github-hosted" ]]; then
        return 0
    else
        return 1
    fi
}

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
    numfmt --to=iec-i --suffix=B --padding=7 "${1}000"
}

# macro to output saved space
printSavedSpace() {
    # Disk space before the operation
    local before=${1}
    local title=${2:-}

    local after
    after=$(getAvailableSpace)
    local saved=$((after - before))

    if [ "$saved" -lt 0 ]; then
        echo "::warning::Saved space is negative: $saved. Using '0' as saved space."
        saved=0
    fi

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

removeUnusedFilesAndDirs() {
    local to_remove=(
        "/usr/share/java"
    )

    if isGitHubRunner; then
        to_remove+=(
            "/usr/local/aws-sam-cli"
            "/usr/local/doc/cmake"
            "/usr/local/julia"*
            "/usr/local/lib/android"
            "/usr/local/share/chromedriver-"*
            "/usr/local/share/chromium"
            "/usr/local/share/cmake-"*
            "/usr/local/share/edge_driver"
            "/usr/local/share/emacs"
            "/usr/local/share/gecko_driver"
            "/usr/local/share/icons"
            "/usr/local/share/powershell"
            "/usr/local/share/vcpkg"
            "/usr/local/share/vim"
            "/usr/share/apache-maven-"*
            "/usr/share/gradle-"*
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
        )

        if [ -n "${AGENT_TOOLSDIRECTORY:-}" ]; then
            # Environment variable set by GitHub Actions
            to_remove+=(
                "${AGENT_TOOLSDIRECTORY}"
            )
        else
            echo "::warning::AGENT_TOOLSDIRECTORY is not set. Skipping removal."
        fi
    else
        # Remove folders and files present in AWS CodeBuild
        to_remove+=(
            # binaries
            "/usr/local/bin/ecs-cli"
            "/usr/local/bin/eksctl"
            "/usr/local/bin/kubectl"

            "${HOME}/.gradle"
            "${HOME}/.dotnet"
            "${HOME}/.goenv"
            "${HOME}/.phpenv"

        )
    fi

    for element in "${to_remove[@]}"; do
        if [ ! -e "$element" ]; then
            # The file or directory doesn't exist.
            # Maybe it was removed in a newer version of the runner or it's not present in a
            # specific architecture (e.g. ARM).
            echo "::warning::Directory or file $element does not exist, skipping."
        fi
    done

    # Remove all files and directories at once to save time.
    sudo rm -rf "${to_remove[@]}"
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
    local packages=(
        '^aspnetcore-.*'
        '^dotnet-.*'
        '^llvm-.*'
        '^mongodb-.*'
        'firefox'
        'libgl1-mesa-dri'
        'mono-devel'
        'php.*'
    )

    if isGitHubRunner; then
        packages+=(
            azure-cli
        )

        if isX86; then
            packages+=(
                'google-chrome-stable'
                'google-cloud-cli'
                'google-cloud-sdk'
                'powershell'
            )
        fi
    else
        packages+=(
            'google-chrome-stable'
        )
    fi

    WAIT_DPKG_LOCK="-o DPkg::Lock::Timeout=60"
    sudo apt-get ${WAIT_DPKG_LOCK} -qq remove -y --fix-missing "${packages[@]}"

    sudo apt-get ${WAIT_DPKG_LOCK} autoremove -y \
        || echo "::warning::The command [sudo apt-get autoremove -y] failed"
    sudo apt-get ${WAIT_DPKG_LOCK} clean \
        || echo "::warning::The command [sudo apt-get clean] failed"
}

# Remove Docker images.
# Ubuntu 22 runners have docker images already installed.
# They aren't present in ubuntu 24 runners.
cleanDocker() {
    echo "=> Removing the following docker images:"
    sudo docker image ls
    echo "=> Removing docker images..."
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
execAndMeasureSpaceChange cleanPackages "Unused packages"
execAndMeasureSpaceChange cleanDocker "Docker images"
execAndMeasureSpaceChange cleanSwap "Swap storage"
execAndMeasureSpaceChange removeUnusedFilesAndDirs "Unused files and directories"

# Output saved space statistic
echo ""
printDF "AFTER CLEAN-UP:"

echo ""
echo ""

printSavedSpace "$AVAILABLE_INITIAL" "Total saved"
