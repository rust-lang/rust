#!/bin/bash --verbose
set -e

SRC_DIR="target/libcore"
rm -rf $SRC_DIR
mkdir -p $SRC_DIR/src
cp -r $(dirname $(rustup which rustc))/../lib/rustlib/src/rust/src/libcore $SRC_DIR/src/libcore || (echo "Please install rust-src component"; exit 1)
cd $SRC_DIR
git init
git add .
git commit -m "Initial commit" -q
git apply ../../000*.patch

echo "Successfully prepared libcore for building"
