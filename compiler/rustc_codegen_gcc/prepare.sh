#!/bin/bash --verbose
set -e

source prepare_build.sh

cargo install hyperfine || echo "Skipping hyperfine install"

git clone https://github.com/rust-lang/regex.git || echo "rust-lang/regex has already been cloned"
pushd regex
git checkout -- .
git checkout 341f207c1071f7290e3f228c710817c280c8dca1
popd

git clone https://github.com/ebobby/simple-raytracer || echo "ebobby/simple-raytracer has already been cloned"
pushd simple-raytracer
git checkout -- .
git checkout 804a7a21b9e673a482797aa289a18ed480e4d813

# build with cg_llvm for perf comparison
cargo build
mv target/debug/main raytracer_cg_llvm
popd
