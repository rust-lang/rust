#!/bin/bash --verbose
set -e

rustup component add rust-src
./build_sysroot/prepare_sysroot_src.sh
cargo install hyperfine || echo "Skipping hyperfine install"
