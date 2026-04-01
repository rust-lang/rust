#!/usr/bin/env bash

set -e
echo "[BUILD] build system" 1>&2
pushd $(dirname "$0")/build_system > /dev/null
cargo build --release
popd > /dev/null
$(dirname "$0")/build_system/target/release/y $@
