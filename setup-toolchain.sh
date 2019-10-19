#!/bin/bash
# Set up the appropriate rustc toolchain

set -e

cd "$(dirname "$0")"

if [[ "$CI" == true ]] || ! command -v rustup-toolchain-install-master > /dev/null; then
  cargo install -Z install-upgrade rustup-toolchain-install-master --bin rustup-toolchain-install-master
fi

rustup-toolchain-install-master -f -n master
rustup override set master
