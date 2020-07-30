#!/bin/bash
set -e
cd $(dirname "$0")

SRC_DIR=$(dirname $(rustup which rustc))"/../lib/rustlib/src/rust/"
DST_DIR="sysroot_src"

if [ ! -e $SRC_DIR ]; then
    echo "Please install rust-src component"
    exit 1
fi

rm -rf $DST_DIR
mkdir -p $DST_DIR/library
cp -r $SRC_DIR/library $DST_DIR/

pushd $DST_DIR
echo "[GIT] init"
git init
echo "[GIT] add"
git add .
echo "[GIT] commit"
git commit -m "Initial commit" -q
for file in $(ls ../../patches/ | grep -v patcha); do
echo "[GIT] apply" $file
git apply ../../patches/$file
git add -A
git commit --no-gpg-sign -m "Patch $file"
done
popd

echo "Successfully prepared libcore for building"
