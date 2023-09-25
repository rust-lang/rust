#!/usr/bin/env bash

set -e
echo "[BUILD] build system" 1>&2
cd build_system
cargo build --release
cd ..
./build_system/target/release/y $@
