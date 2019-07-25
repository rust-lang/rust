#!/bin/bash
source config.sh

rm -r target/out || true
mkdir -p target/out/clif

echo "[BUILD] sysroot"
time ./build_sysroot/build_sysroot.sh

echo "[BUILD+RUN] alloc_example"
$RUSTC example/alloc_example.rs --crate-type bin
./target/out/alloc_example
