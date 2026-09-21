#!/bin/bash

set -eux

target="${1}"

# Allow setting a channel to account for required components (MinGW)
channel="${2:-nightly}"

# Enable 32-bit tools if needed. Note this must come after existing path entries
# like `/c/mingw64/bin`, otherwise they will be preferred over the host tools.
if [ "$target" = "i686-pc-windows-gnu" ]; then
    PATH="$PATH:/mingw32/bin"
    # Persist this for future jobs
    echo "PATH=$PATH" >> "$GITHUB_ENV"
fi

# Some runners (native ppc and s390x, self-hosted) don't have all the dependencies
# we need, so we need to install them.

to_install=()

if [ "$RUN_IN_DOCKER" != "0" ]; then
    ! command -v rustup && to_install+=(rustup)
    ! command -v m4 && to_install+=(m4)
fi

if [ "$target" = "i686-pc-windows-gnu" ]; then
    ! command -v i686-w64-mingw32-gcc && to_install+=(mingw-w64-i686-gcc)
fi

if [ ${#to_install[@]} -ne 0 ]; then
    if command -v apt-get; then
        sudo apt-get update
        sudo apt-get install -y "${to_install[@]}"
    elif command -v apk; then
        doas apk add "${to_install[@]}"
    elif command -v pacman; then
        pacman -S --noconfirm "${to_install[@]}"
    else
        echo "No package manager found"
    fi
fi

# Install the correct Rust version
rustup update "$channel" --no-self-update
rustup default "$channel"
rustup target add "$target"
