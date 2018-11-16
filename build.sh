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

build_example_bin() {
    $RUSTC $2 --crate-name $1 --crate-type bin
    sh -c ./target/out/$1 || true
}

if [[ "$1" == "--release" ]]; then
    channel='release'
    cargo build --release
else
    channel='debug'
    cargo build
fi

export RUSTFLAGS='-Zalways-encode-mir -Cpanic=abort -Zcodegen-backend='$(pwd)'/target/'$channel'/librustc_codegen_cranelift.'$dylib_ext
export XARGO_RUST_SRC=$(pwd)'/target/libcore/src'
RUSTC="rustc $RUSTFLAGS -L crate=target/out --out-dir target/out"

rm -r target/out || true
mkdir -p target/out/clif

echo "[BUILD] mini_core"
$RUSTC example/mini_core.rs --crate-name mini_core --crate-type lib

echo "[BUILD] example"
$RUSTC example/example.rs --crate-type lib

echo "[JIT] mini_core_hello_world"
SHOULD_RUN=1 $RUSTC --crate-type bin example/mini_core_hello_world.rs --cfg jit

echo "[AOT] mini_core_hello_world"
build_example_bin mini_core_hello_world example/mini_core_hello_world.rs

pushd xargo
rm -r ~/.xargo/HOST || true
rm -r target || true
time xargo build
popd

# TODO linux linker doesn't accept duplicate definitions
#$RUSTC --sysroot ~/.xargo/HOST example/alloc_example.rs --crate-type bin
#./target/out/alloc_example

$RUSTC --sysroot ~/.xargo/HOST example/mod_bench.rs --crate-type bin

echo "[BUILD] RUSTFLAGS=-Zmir-opt-level=3"
pushd xargo
rm -r ~/.xargo/HOST || true
rm -r target || true
time RUSTFLAGS="-Zmir-opt-level=3 $RUSTFLAGS" xargo build
popd

COMPILE_MOD_BENCH_INLINE="$RUSTC --sysroot ~/.xargo/HOST example/mod_bench.rs --crate-type bin -Zmir-opt-level=3 -Og --crate-name mod_bench_inline"
COMPILE_MOD_BENCH_LLVM_0="rustc example/mod_bench.rs --crate-type bin -Copt-level=0 -o target/out/mod_bench_llvm_0 -Cpanic=abort"
COMPILE_MOD_BENCH_LLVM_1="rustc example/mod_bench.rs --crate-type bin -Copt-level=1 -o target/out/mod_bench_llvm_1 -Cpanic=abort"
COMPILE_MOD_BENCH_LLVM_2="rustc example/mod_bench.rs --crate-type bin -Copt-level=2 -o target/out/mod_bench_llvm_2 -Cpanic=abort"
COMPILE_MOD_BENCH_LLVM_3="rustc example/mod_bench.rs --crate-type bin -Copt-level=3 -o target/out/mod_bench_llvm_3 -Cpanic=abort"

# Use 100 runs, because a single compilations doesn't take more than ~150ms, so it isn't very slow
hyperfine --runs 100 "$COMPILE_MOD_BENCH_INLINE" "$COMPILE_MOD_BENCH_LLVM_0" "$COMPILE_MOD_BENCH_LLVM_1" "$COMPILE_MOD_BENCH_LLVM_2" "$COMPILE_MOD_BENCH_LLVM_3"

echo
echo "[Bench] mod_bench"
hyperfine ./target/out/mod_bench{,_inline} ./target/out/mod_bench_llvm_*

cat target/out/log.txt | sort | uniq -c
