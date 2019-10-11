#!/bin/bash
# Set up the appropriate rustc toolchain

cd "$(dirname "$0")" || exit

if ! command -v rustup-toolchain-install-master > /dev/null; then
  cargo install rustup-toolchain-install-master --debug
fi

rustup-toolchain-install-master -f -n master -c rustc-dev
rustup override set master
