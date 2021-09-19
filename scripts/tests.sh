#!/usr/bin/env bash

set -e

source scripts/config.sh
source scripts/ext_config.sh
export RUSTC=false # ensure that cg_llvm isn't accidentally used
MY_RUSTC="$(pwd)/build/bin/cg_clif $RUSTFLAGS -L crate=target/out --out-dir target/out -Cdebuginfo=2"

function no_sysroot_tests() {
    echo "[BUILD] mini_core"
    $MY_RUSTC example/mini_core.rs --crate-name mini_core --crate-type lib,dylib --target "$TARGET_TRIPLE"

    echo "[BUILD] example"
    $MY_RUSTC example/example.rs --crate-type lib --target "$TARGET_TRIPLE"

    if [[ "$JIT_SUPPORTED" = "1" ]]; then
        echo "[JIT] mini_core_hello_world"
        CG_CLIF_JIT_ARGS="abc bcd" $MY_RUSTC -Zunstable-options -Cllvm-args=mode=jit -Cprefer-dynamic example/mini_core_hello_world.rs --cfg jit --target "$HOST_TRIPLE"

        echo "[JIT-lazy] mini_core_hello_world"
        CG_CLIF_JIT_ARGS="abc bcd" $MY_RUSTC -Zunstable-options -Cllvm-args=mode=jit-lazy -Cprefer-dynamic example/mini_core_hello_world.rs --cfg jit --target "$HOST_TRIPLE"
    else
        echo "[JIT] mini_core_hello_world (skipped)"
    fi

    echo "[AOT] mini_core_hello_world"
    $MY_RUSTC example/mini_core_hello_world.rs --crate-name mini_core_hello_world --crate-type bin -g --target "$TARGET_TRIPLE"
    $RUN_WRAPPER ./target/out/mini_core_hello_world abc bcd
    # (echo "break set -n main"; echo "run"; sleep 1; echo "si -c 10"; sleep 1; echo "frame variable") | lldb -- ./target/out/mini_core_hello_world abc bcd
}

function base_sysroot_tests() {
    echo "[AOT] arbitrary_self_types_pointers_and_wrappers"
    $MY_RUSTC example/arbitrary_self_types_pointers_and_wrappers.rs --crate-name arbitrary_self_types_pointers_and_wrappers --crate-type bin --target "$TARGET_TRIPLE"
    $RUN_WRAPPER ./target/out/arbitrary_self_types_pointers_and_wrappers

    echo "[AOT] alloc_system"
    $MY_RUSTC example/alloc_system.rs --crate-type lib --target "$TARGET_TRIPLE"

    echo "[AOT] alloc_example"
    $MY_RUSTC example/alloc_example.rs --crate-type bin --target "$TARGET_TRIPLE"
    $RUN_WRAPPER ./target/out/alloc_example

    if [[ "$JIT_SUPPORTED" = "1" ]]; then
        echo "[JIT] std_example"
        $MY_RUSTC -Zunstable-options -Cllvm-args=mode=jit -Cprefer-dynamic example/std_example.rs --target "$HOST_TRIPLE"

        echo "[JIT-lazy] std_example"
        $MY_RUSTC -Zunstable-options -Cllvm-args=mode=jit-lazy -Cprefer-dynamic example/std_example.rs --target "$HOST_TRIPLE"
    else
        echo "[JIT] std_example (skipped)"
    fi

    echo "[AOT] dst_field_align"
    # FIXME Re-add -Zmir-opt-level=2 once rust-lang/rust#67529 is fixed.
    $MY_RUSTC example/dst-field-align.rs --crate-name dst_field_align --crate-type bin --target "$TARGET_TRIPLE"
    $RUN_WRAPPER ./target/out/dst_field_align || (echo $?; false)

    echo "[AOT] std_example"
    $MY_RUSTC example/std_example.rs --crate-type bin --target "$TARGET_TRIPLE"
    $RUN_WRAPPER ./target/out/std_example arg

    echo "[AOT] subslice-patterns-const-eval"
    $MY_RUSTC example/subslice-patterns-const-eval.rs --crate-type bin -Cpanic=abort --target "$TARGET_TRIPLE"
    $RUN_WRAPPER ./target/out/subslice-patterns-const-eval

    echo "[AOT] track-caller-attribute"
    $MY_RUSTC example/track-caller-attribute.rs --crate-type bin -Cpanic=abort --target "$TARGET_TRIPLE"
    $RUN_WRAPPER ./target/out/track-caller-attribute

    echo "[AOT] mod_bench"
    $MY_RUSTC example/mod_bench.rs --crate-type bin --target "$TARGET_TRIPLE"
    $RUN_WRAPPER ./target/out/mod_bench
}

function extended_sysroot_tests() {
    pushd rand
    ../build/cargo clean
    if [[ "$HOST_TRIPLE" = "$TARGET_TRIPLE" ]]; then
        echo "[TEST] rust-random/rand"
        ../build/cargo test --workspace
    else
        echo "[AOT] rust-random/rand"
        ../build/cargo build --workspace --target $TARGET_TRIPLE --tests
    fi
    popd

    pushd simple-raytracer
    if [[ "$HOST_TRIPLE" = "$TARGET_TRIPLE" ]]; then
        echo "[BENCH COMPILE] ebobby/simple-raytracer"
        hyperfine --runs "${RUN_RUNS:-10}" --warmup 1 --prepare "../build/cargo clean" \
        "RUSTC=rustc RUSTFLAGS='' cargo build" \
        "../build/cargo build"

        echo "[BENCH RUN] ebobby/simple-raytracer"
        cp ./target/debug/main ./raytracer_cg_clif
        hyperfine --runs "${RUN_RUNS:-10}" ./raytracer_cg_llvm ./raytracer_cg_clif
    else
        ../build/cargo clean
        echo "[BENCH COMPILE] ebobby/simple-raytracer (skipped)"
        echo "[COMPILE] ebobby/simple-raytracer"
        ../build/cargo build --target $TARGET_TRIPLE
        echo "[BENCH RUN] ebobby/simple-raytracer (skipped)"
    fi
    popd

    pushd build_sysroot/sysroot_src/library/core/tests
    echo "[TEST] libcore"
    ../../../../../build/cargo clean
    if [[ "$HOST_TRIPLE" = "$TARGET_TRIPLE" ]]; then
        ../../../../../build/cargo test
    else
        ../../../../../build/cargo build --target $TARGET_TRIPLE --tests
    fi
    popd

    pushd regex
    echo "[TEST] rust-lang/regex example shootout-regex-dna"
    ../build/cargo clean
    export RUSTFLAGS="$RUSTFLAGS --cap-lints warn" # newer aho_corasick versions throw a deprecation warning
    # Make sure `[codegen mono items] start` doesn't poison the diff
    ../build/cargo build --example shootout-regex-dna --target $TARGET_TRIPLE
    if [[ "$HOST_TRIPLE" = "$TARGET_TRIPLE" ]]; then
        cat examples/regexdna-input.txt \
            | ../build/cargo run --example shootout-regex-dna --target $TARGET_TRIPLE \
            | grep -v "Spawned thread" > res.txt
        diff -u res.txt examples/regexdna-output.txt
    fi

    if [[ "$HOST_TRIPLE" = "$TARGET_TRIPLE" ]]; then
        echo "[TEST] rust-lang/regex tests"
        ../build/cargo test --tests -- --exclude-should-panic --test-threads 1 -Zunstable-options -q
    else
        echo "[AOT] rust-lang/regex tests"
        ../build/cargo build --tests --target $TARGET_TRIPLE
    fi
    popd

    pushd portable-simd
    echo "[TEST] rust-lang/portable-simd"
    ../build/cargo clean
    ../build/cargo build --all-targets --target $TARGET_TRIPLE
    if [[ "$HOST_TRIPLE" = "$TARGET_TRIPLE" ]]; then
        ../build/cargo test -q
    fi
    popd
}

case "$1" in
    "no_sysroot")
        no_sysroot_tests
        ;;
    "base_sysroot")
        base_sysroot_tests
        ;;
    "extended_sysroot")
        extended_sysroot_tests
        ;;
    *)
        echo "unknown test suite"
        ;;
esac
