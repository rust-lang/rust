#!/bin/bash --verbose
set -e

rustup component add rust-src
./build_sysroot/prepare_sysroot_src.sh
cargo install hyperfine || echo "Skipping hyperfine install"

git clone https://github.com/rust-lang/regex.git || echo "rust-lang/regex has already been cloned"
pushd regex
git checkout -- .
git checkout 341f207c1071f7290e3f228c710817c280c8dca1
git apply ../crate_patches/regex.patch
popd
