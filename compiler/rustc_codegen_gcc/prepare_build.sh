#!/bin/bash --verbose
set -e

rustup component add rust-src rustc-dev llvm-tools-preview
./build_sysroot/prepare_sysroot_src.sh
