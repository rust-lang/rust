#!/bin/bash --verbose
SRC_DIR="target/libcore"
rm -rf $SRC_DIR &&
mkdir -p $SRC_DIR/src &&
cp -r $(dirname $(rustup which rustc))/../lib/rustlib/src/rust/src/libcore $SRC_DIR/src/libcore || (echo "Please install rust-src component"; exit 1)
cd $SRC_DIR || exit 1
git init || exit 1
git add . || exit 1
git commit -m "Initial commit" -q || exit 1
git apply ../../000*.patch || exit 1
echo "Successfully prepared libcore for building"
