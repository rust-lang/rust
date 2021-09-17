#!/bin/bash

#set -x
set -e

if [ -f ./gcc_path ]; then 
    export GCC_PATH=$(cat gcc_path)
else
    echo 'Please put the path to your custom build of libgccjit in the file `gcc_path`, see Readme.md for details'
    exit 1
fi

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
