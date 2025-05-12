#!/bin/bash
# Looks like docker containers have IPv6 disabled by default, so let's turn it
# on since libstd tests require it

set -euo pipefail
IFS=$'\n\t'

source "$(cd "$(dirname "$0")" && pwd)/../shared.sh"

if isLinux; then
    sudo mkdir -p /etc/docker
    echo '{"ipv6":true,"fixed-cidr-v6":"fd9a:8454:6789:13f7::/64"}' \
        | sudo tee /etc/docker/daemon.json
    sudo service docker restart
fi
