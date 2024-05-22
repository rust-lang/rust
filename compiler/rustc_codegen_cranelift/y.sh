#!/usr/bin/env bash

set -e
echo "[BUILD] build system" 1>&2
exec cargo run --manifest-path build_system/Cargo.toml -- "$@"
