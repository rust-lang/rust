#!/usr/bin/env bash

set -e
echo "[BUILD] build system" 1>&2
mkdir -p build_system/target
rustc build_system/src/main.rs -o build_system/target/y -Cdebuginfo=1 --edition 2021
exec ./build_system/target/y "$@"
