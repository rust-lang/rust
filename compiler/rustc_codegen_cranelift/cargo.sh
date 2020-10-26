#!/bin/bash

if [ -z $CHANNEL ]; then
export CHANNEL='release'
fi

pushd $(dirname "$0") >/dev/null
source scripts/config.sh

# read nightly compiler from rust-toolchain file
TOOLCHAIN=$(cat rust-toolchain)

popd >/dev/null

cmd=$1
shift

if [[ "$cmd" = "jit" ]]; then
cargo +${TOOLCHAIN} rustc $@ -- --jit
else
cargo +${TOOLCHAIN} $cmd $@
fi
