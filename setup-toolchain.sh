#!/bin/bash
# Set up the appropriate rustc toolchain

set -e

cd "$(dirname "$0")"

ERRNO=0
RTIM_PATH=$(command -v rustup-toolchain-install-master) || ERRNO=$?
CARGO_HOME=${CARGO_HOME:-$HOME/.cargo}

# Check if people also install RTIM in other locations beside
# ~/.cargo/bin
if [[ "$ERRNO" -ne 0 ]] || [[ "$RTIM_PATH" == $CARGO_HOME/bin/rustup-toolchain-install-master ]]; then
    cargo install -Z install-upgrade rustup-toolchain-install-master
else
    VERSION=$(rustup-toolchain-install-master -V | grep -o "[0-9.]*")
    REMOTE=$(cargo search rustup-toolchain-install-master | grep -o "[0-9.]*")
    echo "info: skipping updating rustup-toolchain-install-master at $RTIM_PATH"
    echo "      current version : $VERSION"
    echo "      remote version  : $REMOTE"
fi

RUST_COMMIT=$(git ls-remote https://github.com/rust-lang/rust master | awk '{print $1}')

if rustc +master -Vv 2>/dev/null | grep -q "$RUST_COMMIT"; then
    echo "info: master toolchain is up-to-date"
    exit 0
fi

rustup-toolchain-install-master -f -n master -c rustc-dev -- "$RUST_COMMIT"
rustup override set master
