#!/usr/bin/env bash

if [ -z $CHANNEL ]; then
export CHANNEL='debug'
fi

pushd $(dirname "$0") >/dev/null
source config.sh

# read nightly compiler from rust-toolchain file
TOOLCHAIN=$(cat rust-toolchain | grep channel | sed 's/channel = "\(.*\)"/\1/')

popd >/dev/null

if [[ $(${RUSTC} -V) != $(${RUSTC} +${TOOLCHAIN} -V) ]]; then
    echo "rustc_codegen_gcc is build for $(rustc +${TOOLCHAIN} -V) but the default rustc version is $(rustc -V)."
    echo "Using $(rustc +${TOOLCHAIN} -V)."
fi

cmd=$1
shift

RUSTDOCFLAGS="$RUSTFLAGS" cargo +${TOOLCHAIN} $cmd $@
