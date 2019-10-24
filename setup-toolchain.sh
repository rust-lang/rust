#!/bin/bash
# Set up the appropriate rustc toolchain

set -e

cd "$(dirname "$0")"

if [[ "$CI" == true ]] || ! command -v rustup-toolchain-install-master > /dev/null; then
    cargo install -Z install-upgrade rustup-toolchain-install-master --bin rustup-toolchain-install-master
fi

RUST_COMMIT=$(git ls-remote https://github.com/rust-lang/rust master | awk '{print $1}')

if rustc +master -Vv 2>/dev/null | grep -q "$RUST_COMMIT"; then
    echo "info: master toolchain is up-to-date"
    exit 0
fi

rustup-toolchain-install-master -f -n master -c rustc-dev -- "$RUST_COMMIT"
rustup override set master
