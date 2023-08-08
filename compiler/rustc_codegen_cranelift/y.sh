#!/usr/bin/env bash

set -e
echo "[BUILD] build system" 1>&2
rustc build_system/main.rs -o y.bin -Cdebuginfo=1 --edition 2021
exec ./y.bin "$@"
