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
    command -v libtool || to_install+=(libtool)
    command -v m4 || to_install+=(m4)
    command -v make || to_install+=(make)
    command -v rustup || to_install+=(rustup)
fi

if [ "$target" = "i686-pc-windows-gnu" ]; then
    command -v i686-w64-mingw32-gcc || to_install+=(mingw-w64-i686-gcc)
fi

# specific check for armv7-unknown-linux-gnueabihf on canonical runners
# which are 32-bit armhf-native but have VMs running over an arm64 kernel with AArch32/EL0 support
if [ "$target" = "armv7-unknown-linux-gnueabihf" ] && [ "${RUN_IN_DOCKER:-}" != "true" ] &&
    [ "$(dpkg --print-architecture 2>/dev/null)" = "arm64" ]; then

    # expose these variables so we can keep the arm64 host toolchain, cross-link with
    # the armhf gcc, and run the test binaries natively in AArch32 mode
    {
        echo "CARGO_TARGET_ARMV7_UNKNOWN_LINUX_GNUEABIHF_LINKER=arm-linux-gnueabihf-gcc"
        echo "CC_armv7_unknown_linux_gnueabihf=arm-linux-gnueabihf-gcc"
        echo "AR_armv7_unknown_linux_gnueabihf=arm-linux-gnueabihf-ar"
    } >> "$GITHUB_ENV"

    # install required tooling to enable cross-compilation
    sudo dpkg --add-architecture armhf
    to_install+=(gcc-arm-linux-gnueabihf) # cross-compiler
    to_install+=(libc6-dev-armhf-cross) # target libc
    to_install+=(libc6:armhf libgcc-s1:armhf) # loaded at runtime
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
rustup set profile minimal
rustup update "$channel" --no-self-update
rustup default "$channel"
rustup target add "$target"
