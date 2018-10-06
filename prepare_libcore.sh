#!/bin/bash --verbose
set -e

SRC_DIR=$(dirname $(rustup which rustc))"/../lib/rustlib/src/rust/"
DST_DIR="target/libcore"

if [ ! -e $SRC_DIR ]; then
    echo "Please install rust-src component"
    exit 1
fi

rm -rf $DST_DIR
mkdir -p $DST_DIR/src
cp -r $SRC_DIR/src $DST_DIR/

cd $DST_DIR
git init
git add .
git commit -m "Initial commit" -q
git apply ../../000*.patch

echo "Successfully prepared libcore for building"
