#!/bin/bash

set -e

unamestr=`uname`
if [[ "$unamestr" == 'Linux' ]]; then
   dylib_ext='so'
elif [[ "$unamestr" == 'Darwin' ]]; then
   dylib_ext='dylib'
else
   echo "Unsupported os"
   exit 1
fi

extract_data() {
    pushd target/out/
    ar x $1 data.o
    chmod +rw data.o
    mv data.o $2
    popd
}

link_and_run() {
    target=$1
    shift
    pushd target/out
    gcc $@ -o $target
    sh -c ./$target || true
    popd
}

build_lib() {
    SHOULD_CODEGEN=1 $RUSTC $2 --crate-name $1 --crate-type lib
    extract_data lib$1.rlib $1.o
}

run_bin() {
    SHOULD_RUN=1 $RUSTC $@ --crate-type bin
}

build_example_bin() {
    $RUSTC $2 --crate-name $1 --crate-type bin
    extract_data $1 $1.o

    link_and_run $1 mini_core.o $1.o
}

if [[ "$1" == "--release" ]]; then
    channel='release'
    cargo build --release
else
    channel='debug'
    cargo build
fi

RUSTC="rustc -Zcodegen-backend=$(pwd)/target/$channel/librustc_codegen_cranelift.$dylib_ext -Cpanic=abort -L crate=target/out --out-dir target/out"

rm -r target/out || true
mkdir -p target/out/clif

echo "[BUILD] mini_core"
build_lib mini_core example/mini_core.rs

$RUSTC example/example.rs --crate-type lib

echo "[JIT] mini_core_hello_world"
run_bin example/mini_core_hello_world.rs --cfg jit

echo "[AOT] mini_core_hello_world"
build_example_bin mini_core_hello_world example/mini_core_hello_world.rs

echo "[BUILD] core"
time $RUSTC target/libcore/src/libcore/lib.rs --crate-type lib --crate-name core -Cincremental=target/incremental_core
cat target/out/log.txt | sort | uniq -c
#extract_data libcore.rlib core.o
