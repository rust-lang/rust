#!/bin/bash
source config.sh

rm -r target/out || true
mkdir -p target/out/clif

echo "[BUILD] mini_core"
$RUSTC example/mini_core.rs --crate-name mini_core --crate-type dylib -Csave-temps

echo "[BUILD] example"
$RUSTC example/example.rs --crate-type lib

echo "[JIT] mini_core_hello_world"
SHOULD_RUN=1 JIT_ARGS="abc bcd" $RUSTC --crate-type bin example/mini_core_hello_world.rs --cfg jit

echo "[AOT] mini_core_hello_world"
$RUSTC example/mini_core_hello_world.rs --crate-name mini_core_hello_world --crate-type bin
./target/out/mini_core_hello_world abc bcd

echo "[AOT] arbitrary_self_types_pointers_and_wrappers"
$RUSTC example/arbitrary_self_types_pointers_and_wrappers.rs --crate-name arbitrary_self_types_pointers_and_wrappers --crate-type bin
./target/out/arbitrary_self_types_pointers_and_wrappers

echo "[BUILD] sysroot"
time ./build_sysroot/build_sysroot.sh

echo "[BUILD+RUN] alloc_example"
$RUSTC example/alloc_example.rs --crate-type bin
./target/out/alloc_example

echo "[BUILD+RUN] std_example"
$RUSTC example/std_example.rs --crate-type bin
./target/out/std_example

echo "[BUILD] mod_bench"
$RUSTC example/mod_bench.rs --crate-type bin

# FIXME linker gives multiple definitions error on Linux
#echo "[BUILD] sysroot in release mode"
#./build_sysroot/build_sysroot.sh --release

COMPILE_MOD_BENCH_INLINE="$RUSTC example/mod_bench.rs --crate-type bin -Zmir-opt-level=3 -O --crate-name mod_bench_inline"
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
