#!/bin/bash

dir=$(dirname "$0")
source "$dir/config.sh"

# read nightly compiler from rust-toolchain file
TOOLCHAIN=$(cat "$dir/rust-toolchain" | grep channel | sed "s/channel = \"\(.*\)\"/\1/")

cmd=$1
shift || true

if [[ "$cmd" = "jit" ]]; then
cargo "+${TOOLCHAIN}" rustc "$@" -- --jit
else
cargo "+${TOOLCHAIN}" "$cmd" "$@"
fi
