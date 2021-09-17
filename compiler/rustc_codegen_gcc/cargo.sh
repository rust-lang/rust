#!/bin/bash

if [ -z $CHANNEL ]; then
export CHANNEL='debug'
fi

pushd $(dirname "$0") >/dev/null
source config.sh

# read nightly compiler from rust-toolchain file
TOOLCHAIN=$(cat rust-toolchain)

popd >/dev/null

if [[ $(rustc -V) != $(rustc +${TOOLCHAIN} -V) ]]; then
    echo "rustc_codegen_gcc is build for $(rustc +${TOOLCHAIN} -V) but the default rustc version is $(rustc -V)."
    echo "Using $(rustc +${TOOLCHAIN} -V)."
fi

cmd=$1
shift

RUSTDOCFLAGS="$RUSTFLAGS" cargo +${TOOLCHAIN} $cmd --target $TARGET_TRIPLE $@
