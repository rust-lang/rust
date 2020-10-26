#!/bin/bash
set -e

# Build cg_clif
export RUSTFLAGS="-Zrun_dsymutil=no"
if [[ "$1" == "--release" ]]; then
    export CHANNEL='release'
    cargo build --release
else
    export CHANNEL='debug'
    cargo build --bin cg_clif
fi

# Config
source scripts/config.sh
export CG_CLIF_INCR_CACHE_DISABLED=1
RUSTC=$RUSTC" "$RUSTFLAGS" -L crate=target/out --out-dir target/out -Cdebuginfo=2"

# Cleanup
rm -r target/out || true

# Perform all tests
echo "[BUILD] mini_core"
$RUSTC example/mini_core.rs --crate-name mini_core --crate-type lib,dylib --target $TARGET_TRIPLE

echo "[BUILD] example"
$RUSTC example/example.rs --crate-type lib --target $TARGET_TRIPLE

if [[ "$JIT_SUPPORTED" = "1" ]]; then
    echo "[JIT] mini_core_hello_world"
    CG_CLIF_JIT_ARGS="abc bcd" $RUSTC --jit example/mini_core_hello_world.rs --cfg jit --target $HOST_TRIPLE
else
    echo "[JIT] mini_core_hello_world (skipped)"
fi

echo "[AOT] mini_core_hello_world"
$RUSTC example/mini_core_hello_world.rs --crate-name mini_core_hello_world --crate-type bin -g --target $TARGET_TRIPLE
$RUN_WRAPPER ./target/out/mini_core_hello_world abc bcd
# (echo "break set -n main"; echo "run"; sleep 1; echo "si -c 10"; sleep 1; echo "frame variable") | lldb -- ./target/out/mini_core_hello_world abc bcd

echo "[AOT] arbitrary_self_types_pointers_and_wrappers"
$RUSTC example/arbitrary_self_types_pointers_and_wrappers.rs --crate-name arbitrary_self_types_pointers_and_wrappers --crate-type bin --target $TARGET_TRIPLE
$RUN_WRAPPER ./target/out/arbitrary_self_types_pointers_and_wrappers

echo "[BUILD] sysroot"
time ./build_sysroot/build_sysroot.sh --release

echo "[AOT] alloc_example"
$RUSTC example/alloc_example.rs --crate-type bin --target $TARGET_TRIPLE
$RUN_WRAPPER ./target/out/alloc_example

if [[ "$JIT_SUPPORTED" = "1" ]]; then
    echo "[JIT] std_example"
    $RUSTC --jit example/std_example.rs --target $HOST_TRIPLE
else
    echo "[JIT] std_example (skipped)"
fi

echo "[AOT] dst_field_align"
# FIXME Re-add -Zmir-opt-level=2 once rust-lang/rust#67529 is fixed.
$RUSTC example/dst-field-align.rs --crate-name dst_field_align --crate-type bin --target $TARGET_TRIPLE
$RUN_WRAPPER ./target/out/dst_field_align || (echo $?; false)

echo "[AOT] std_example"
$RUSTC example/std_example.rs --crate-type bin --target $TARGET_TRIPLE
$RUN_WRAPPER ./target/out/std_example arg

echo "[AOT] subslice-patterns-const-eval"
$RUSTC example/subslice-patterns-const-eval.rs --crate-type bin -Cpanic=abort --target $TARGET_TRIPLE
$RUN_WRAPPER ./target/out/subslice-patterns-const-eval

echo "[AOT] track-caller-attribute"
$RUSTC example/track-caller-attribute.rs --crate-type bin -Cpanic=abort --target $TARGET_TRIPLE
$RUN_WRAPPER ./target/out/track-caller-attribute

echo "[AOT] mod_bench"
$RUSTC example/mod_bench.rs --crate-type bin --target $TARGET_TRIPLE
$RUN_WRAPPER ./target/out/mod_bench

pushd rand
rm -r ./target || true
../cargo.sh test --workspace
popd

pushd simple-raytracer
if [[ "$HOST_TRIPLE" = "$TARGET_TRIPLE" ]]; then
    echo "[BENCH COMPILE] ebobby/simple-raytracer"
    hyperfine --runs ${RUN_RUNS:-10} --warmup 1 --prepare "cargo clean" \
    "RUSTC=rustc RUSTFLAGS='' cargo build" \
    "../cargo.sh build"

    echo "[BENCH RUN] ebobby/simple-raytracer"
    cp ./target/debug/main ./raytracer_cg_clif
    hyperfine --runs ${RUN_RUNS:-10} ./raytracer_cg_llvm ./raytracer_cg_clif
else
    echo "[BENCH COMPILE] ebobby/simple-raytracer (skipped)"
    echo "[COMPILE] ebobby/simple-raytracer"
    ../cargo.sh build
    echo "[BENCH RUN] ebobby/simple-raytracer (skipped)"
fi
popd

pushd build_sysroot/sysroot_src/library/core/tests
echo "[TEST] libcore"
rm -r ./target || true
../../../../../cargo.sh test
popd

pushd regex
echo "[TEST] rust-lang/regex example shootout-regex-dna"
../cargo.sh clean
# Make sure `[codegen mono items] start` doesn't poison the diff
../cargo.sh build --example shootout-regex-dna
cat examples/regexdna-input.txt | ../cargo.sh run --example shootout-regex-dna | grep -v "Spawned thread" > res.txt
diff -u res.txt examples/regexdna-output.txt

echo "[TEST] rust-lang/regex tests"
../cargo.sh test --tests -- --exclude-should-panic --test-threads 1 -Zunstable-options
popd
