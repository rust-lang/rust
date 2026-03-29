#!/bin/bash

set -eux

python3 ../x.py dist \
    --host $HOSTS --target $HOSTS \
    --include-default-paths \
    build-manifest bootstrap

find / -type f -name rust-src-nightly.tar.xz
which generate || echo "generate not found"

# ./x doesn't have a way to build an rlib-only std but cargo's build-std does.
# So let's try that
RUSTC=$(realpath -s ./build/$HOSTS/stage2/bin/rustc)
CARGO=$(realpath -s ./build/$HOSTS/stage2-tools-bin/cargo)
lib_name=be05c8a7_f3cd_4df9_a9b5_65fee92c7db6

$CARGO new --lib $lib_name
cd $lib_name
truncate -s 0 src/lib.rs
RUSTC_BOOTSTRAP=1 $CARGO build --target $DIST_TARGETS -Zbuild-std --release

cd ..
unset CARGO
unset RUSTC

cd build
rlib_path=rust-std-$DIST_TARGETS/lib/rustlib/$DIST_TARGETS/lib
mkdir -p $rlib_path/self-contained
cp -r ../$lib_name/target/$DIST_TARGETS/release/deps/. rlib_path
rm rlib_path/*$lib_name*
cd rust-std-$DIST_TARGETS
find lib -type f -printf 'file:%p\n' > manifest.in
cd ../..
tar -cJvf ./build/dist/rust-std-$DIST_TARGETS.xz ./build/rust-std-$DIST_TARGETS
