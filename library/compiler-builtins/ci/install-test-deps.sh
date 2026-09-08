#!/bin/bash

set -eux

target="${1}"

# Allow setting a channel to account for required components (MinGW)
channel="${2:-nightly}"

# Some runners (native ppc and s390x, self-hosted) don't have all the dependencies
# we need, so we need to install them.

needed_deps=()
to_install=()

if [ "$RUN_IN_DOCKER" != "0" ]; then
    needed_deps+=(rustup m4)
fi

for dep in "${needed_deps[@]}"; do
    ! command -v "$dep" && to_install+=("$dep")
done

if [ ${#to_install[@]} -ne 0 ]; then
    if command -v apt-get; then
        sudo apt-get update
        sudo apt-get install -y "${to_install[@]}"
    elif command -v apk; then
        doas apk add "${to_install[@]}"
    else
        echo "No package manager found"
    fi
fi

# Install the correct Rust version
rustup update "$channel" --no-self-update
rustup default "$channel"
rustup target add "$target"
