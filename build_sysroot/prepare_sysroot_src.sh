#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

SRC_DIR="$(dirname "$(rustup which rustc)")/../lib/rustlib/src/rust/"
DST_DIR="sysroot_src"

if [ ! -e "$SRC_DIR" ]; then
    echo "Please install rust-src component"
    exit 1
fi

rm -rf $DST_DIR
mkdir -p $DST_DIR/library
cp -a "$SRC_DIR/library" $DST_DIR/

pushd $DST_DIR
echo "[GIT] init"
git init
echo "[GIT] add"
git add .
echo "[GIT] commit"
git commit -m "Initial commit" -q
for file in $(ls ../../patches/ | grep -v patcha); do
echo "[GIT] apply" "$file"
git apply ../../patches/"$file"
git add -A
git commit --no-gpg-sign -m "Patch $file"
done
popd

git clone https://github.com/rust-lang/compiler-builtins.git || echo "rust-lang/compiler-builtins has already been cloned"
pushd compiler-builtins
git checkout -- .
git checkout 0.1.39
git apply ../../crate_patches/000*-compiler-builtins-*.patch
popd

echo "Successfully prepared sysroot source for building"
