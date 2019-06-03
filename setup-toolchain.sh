#!/bin/bash
# Set up the appropriate rustc toolchain

cd "$(dirname "$0")"

if ! command -v rustup-toolchain-install-master > /dev/null; then
  cargo install rustup-toolchain-install-master --debug
fi

RUSTC_HASH=$(git ls-remote https://github.com/rust-lang/rust.git master | awk '{print $1}')
rustup-toolchain-install-master -f -n master "$RUSTC_HASH"
rustup override set master
