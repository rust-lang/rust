#!/bin/bash

#set -x
set -e

export GCC_PATH=$(cat gcc_path)

export LD_LIBRARY_PATH="$GCC_PATH"
export LIBRARY_PATH="$GCC_PATH"

if [[ "$1" == "--release" ]]; then
    export CHANNEL='release'
    CARGO_INCREMENTAL=1 cargo rustc --release
else
    echo $LD_LIBRARY_PATH
    export CHANNEL='debug'
    cargo rustc
fi

source config.sh

rm -r target/out || true
mkdir -p target/out/gccjit

echo "[BUILD] sysroot"
time ./build_sysroot/build_sysroot.sh $CHANNEL
