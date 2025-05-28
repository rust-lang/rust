#!/bin/bash
# Update Docker to the latest version on Ubuntu

set -euo pipefail

echo "previous docker version:"
docker --version || true

# Remove old Docker packages
for pkg in \
    docker.io \
    docker-compose \
    docker-compose-v2 \
    docker-doc \
    podman-docker ;
    do sudo apt-get remove -y $pkg || true; done

sudo apt-get update

# Add Docker's official GPG key
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Set up the repository
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update

sudo apt-get install -y \
  docker-ce \
  docker-ce-cli \
  containerd.io \
  docker-buildx-plugin

if ! docker --version; then
    echo "Docker installation failed"
    exit 1
fi

echo "Docker updated successfully! New version:"
docker --version
# # Start and enable Docker service
# sudo systemctl start docker
# sudo systemctl enable docker

# # Add current user to docker group (if not already)
# sudo usermod -aG docker $USER || true
