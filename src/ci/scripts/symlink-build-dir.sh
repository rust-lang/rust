#!/bin/bash
# We've had multiple issues with the default disk running out of disk space
# during builds, and it looks like other disks mounted in the VMs have more
# space available. This script synlinks the build directory to those other
# disks, in the CI providers and OSes affected by this.

set -euo pipefail
IFS=$'\n\t'

source "$(cd "$(dirname "$0")" && pwd)/../shared.sh"

if isWindows && isAzurePipelines; then
    cmd //c "mkdir c:\\MORE_SPACE"
    cmd //c "mklink /J build c:\\MORE_SPACE"
elif isLinux && isGitHubActions && ! isSelfHostedGitHubActions; then
    sudo mkdir -p /mnt/more-space
    sudo chown -R "$(whoami):" /mnt/more-space

    # Switch the whole workspace to the /mnt partition, which has more space.
    # We don't just symlink the `obj` directory as doing that creates problems
    # with the docker container.
    current_dir="$(readlink -f "$(pwd)")"
    cd /tmp
    mv "${current_dir}" /mnt/more-space/workspace
    ln -s /mnt/more-space/workspace "${current_dir}"
    cd "${current_dir}"

    # Move the Docker data directory to /mnt
    sudo systemctl stop docker.service
    sudo mv /var/lib/docker /mnt/docker
    sudo ln -s /mnt/docker /var/lib/docker
    sudo systemctl start docker.service
fi
