#!/usr/bin/env bash
# Set up the appropriate rustc toolchain

set -e

cd "$(dirname "$0")"

RTIM_PATH=$(command -v rustup-toolchain-install-master) || INSTALLED=false
CARGO_HOME=${CARGO_HOME:-$HOME/.cargo}

# Check if RTIM is not installed or installed in other locations not in ~/.cargo/bin
if [[ "$INSTALLED" == false || "$RTIM_PATH" == $CARGO_HOME/bin/rustup-toolchain-install-master ]]; then
    cargo +nightly install rustup-toolchain-install-master
else
    VERSION=$(rustup-toolchain-install-master -V | grep -o "[0-9.]*")
    REMOTE=$(cargo +nightly search rustup-toolchain-install-master | grep -o "[0-9.]*")
    echo "info: skipping updating rustup-toolchain-install-master at $RTIM_PATH"
    echo "      current version : $VERSION"
    echo "      remote version  : $REMOTE"
fi

RUST_COMMIT=$(git ls-remote https://github.com/rust-lang/rust master | awk '{print $1}')

if rustc +master -Vv 2>/dev/null | grep -q "$RUST_COMMIT"; then
    echo "info: master toolchain is up-to-date"
    exit 0
fi

if [[ -n "$HOST_TOOLCHAIN" ]]; then
    TOOLCHAIN=('--host' "$HOST_TOOLCHAIN")
else
    TOOLCHAIN=()
fi

rustup-toolchain-install-master -f -n master "${TOOLCHAIN[@]}" -c rustc-dev -- "$RUST_COMMIT"
rustup override set master
