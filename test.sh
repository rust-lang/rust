#!/bin/bash
source config.sh

build_example_bin() {
    $RUSTC $2 --crate-name $1 --crate-type bin
    sh -c ./target/out/$1 || true
}

rm -r target/out || true
mkdir -p target/out/clif

echo "[BUILD] mini_core"
$RUSTC example/mini_core.rs --crate-name mini_core --crate-type lib -g

echo "[BUILD] example"
$RUSTC example/example.rs --crate-type lib -g

echo "[JIT] mini_core_hello_world"
SHOULD_RUN=1 $RUSTC --crate-type bin example/mini_core_hello_world.rs --cfg jit

echo "[AOT] mini_core_hello_world"
$RUSTC example/mini_core_hello_world.rs --crate-name mini_core_hello_world --crate-type bin -g
sh -c ./target/out/mini_core_hello_world || true

echo "[BUILD] sysroot"
time ./build_sysroot/build_sysroot.sh

# TODO linux linker doesn't accept duplicate definitions
# echo "[BUILD+RUN] alloc_example"
#$RUSTC --sysroot ./build_sysroot/sysroot example/alloc_example.rs --crate-type bin
#./target/out/alloc_example

echo "[BUILD] mod_bench"
$RUSTC --sysroot ./build_sysroot/sysroot example/mod_bench.rs --crate-type bin -g

echo "[BUILD] sysroot in release mode"
./build_sysroot/build_sysroot.sh --release

COMPILE_MOD_BENCH_INLINE="$RUSTC --sysroot ./build_sysroot/sysroot example/mod_bench.rs --crate-type bin -Zmir-opt-level=3 -Og --crate-name mod_bench_inline"
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
