#!/usr/bin/env bash

set -e

export CG_CLIF_DISPLAY_CG_TIME=1
export CG_CLIF_DISABLE_INCR_CACHE=1

export HOST_TRIPLE=$(rustc -vV | grep host | cut -d: -f2 | tr -d " ")
export TARGET_TRIPLE=${TARGET_TRIPLE:-$HOST_TRIPLE}

export RUN_WRAPPER=''

case "$TARGET_TRIPLE" in
   x86_64*)
      export JIT_SUPPORTED=1
      ;;
   *)
      export JIT_SUPPORTED=0
      ;;
esac

if [[ "$HOST_TRIPLE" != "$TARGET_TRIPLE" ]]; then
   export JIT_SUPPORTED=0
   if [[ "$TARGET_TRIPLE" == "aarch64-unknown-linux-gnu" ]]; then
      # We are cross-compiling for aarch64. Use the correct linker and run tests in qemu.
      export RUSTFLAGS='-Clinker=aarch64-linux-gnu-gcc '$RUSTFLAGS
      export RUN_WRAPPER='qemu-aarch64 -L /usr/aarch64-linux-gnu'
   elif [[ "$TARGET_TRIPLE" == "x86_64-pc-windows-gnu" ]]; then
      # We are cross-compiling for Windows. Run tests in wine.
      export RUN_WRAPPER='wine'
   else
      echo "Unknown non-native platform"
   fi
fi

# FIXME fix `#[linkage = "extern_weak"]` without this
if [[ "$(uname)" == 'Darwin' ]]; then
   export RUSTFLAGS="$RUSTFLAGS -Clink-arg=-undefined -Clink-arg=dynamic_lookup"
fi

MY_RUSTC="$(pwd)/build/rustc-clif $RUSTFLAGS -L crate=target/out --out-dir target/out -Cdebuginfo=2"

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

    echo "[AOT] issue_91827_extern_types"
    $MY_RUSTC example/issue-91827-extern-types.rs --crate-name issue_91827_extern_types --crate-type bin --target "$TARGET_TRIPLE"
    $RUN_WRAPPER ./target/out/issue_91827_extern_types

    echo "[BUILD] alloc_system"
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

    echo "[AOT] std_example"
    $MY_RUSTC example/std_example.rs --crate-type bin --target "$TARGET_TRIPLE"
    $RUN_WRAPPER ./target/out/std_example arg

    echo "[AOT] dst_field_align"
    $MY_RUSTC example/dst-field-align.rs --crate-name dst_field_align --crate-type bin --target "$TARGET_TRIPLE"
    $RUN_WRAPPER ./target/out/dst_field_align

    echo "[AOT] subslice-patterns-const-eval"
    $MY_RUSTC example/subslice-patterns-const-eval.rs --crate-type bin -Cpanic=abort --target "$TARGET_TRIPLE"
    $RUN_WRAPPER ./target/out/subslice-patterns-const-eval

    echo "[AOT] track-caller-attribute"
    $MY_RUSTC example/track-caller-attribute.rs --crate-type bin -Cpanic=abort --target "$TARGET_TRIPLE"
    $RUN_WRAPPER ./target/out/track-caller-attribute

    echo "[AOT] float-minmax-pass"
    $MY_RUSTC example/float-minmax-pass.rs --crate-type bin -Cpanic=abort --target "$TARGET_TRIPLE"
    $RUN_WRAPPER ./target/out/float-minmax-pass

    echo "[AOT] mod_bench"
    $MY_RUSTC example/mod_bench.rs --crate-type bin --target "$TARGET_TRIPLE"
    $RUN_WRAPPER ./target/out/mod_bench
}

function extended_sysroot_tests() {
    pushd rand
    ../build/cargo-clif clean
    if [[ "$HOST_TRIPLE" = "$TARGET_TRIPLE" ]]; then
        echo "[TEST] rust-random/rand"
        ../build/cargo-clif test --workspace
    else
        echo "[AOT] rust-random/rand"
        ../build/cargo-clif build --workspace --target $TARGET_TRIPLE --tests
    fi
    popd

    pushd simple-raytracer
    if [[ "$HOST_TRIPLE" = "$TARGET_TRIPLE" ]]; then
        echo "[BENCH COMPILE] ebobby/simple-raytracer"
        hyperfine --runs "${RUN_RUNS:-10}" --warmup 1 --prepare "../build/cargo-clif clean" \
        "RUSTFLAGS='' cargo build" \
        "../build/cargo-clif build"

        echo "[BENCH RUN] ebobby/simple-raytracer"
        cp ./target/debug/main ./raytracer_cg_clif
        hyperfine --runs "${RUN_RUNS:-10}" ./raytracer_cg_llvm ./raytracer_cg_clif
    else
        ../build/cargo-clif clean
        echo "[BENCH COMPILE] ebobby/simple-raytracer (skipped)"
        echo "[COMPILE] ebobby/simple-raytracer"
        ../build/cargo-clif build --target $TARGET_TRIPLE
        echo "[BENCH RUN] ebobby/simple-raytracer (skipped)"
    fi
    popd

    pushd build_sysroot/sysroot_src/library/core/tests
    echo "[TEST] libcore"
    ../../../../../build/cargo-clif clean
    if [[ "$HOST_TRIPLE" = "$TARGET_TRIPLE" ]]; then
        ../../../../../build/cargo-clif test
    else
        ../../../../../build/cargo-clif build --target $TARGET_TRIPLE --tests
    fi
    popd

    pushd regex
    echo "[TEST] rust-lang/regex example shootout-regex-dna"
    ../build/cargo-clif clean
    export RUSTFLAGS="$RUSTFLAGS --cap-lints warn" # newer aho_corasick versions throw a deprecation warning
    # Make sure `[codegen mono items] start` doesn't poison the diff
    ../build/cargo-clif build --example shootout-regex-dna --target $TARGET_TRIPLE
    if [[ "$HOST_TRIPLE" = "$TARGET_TRIPLE" ]]; then
        cat examples/regexdna-input.txt \
            | ../build/cargo-clif run --example shootout-regex-dna --target $TARGET_TRIPLE \
            | grep -v "Spawned thread" > res.txt
        diff -u res.txt examples/regexdna-output.txt
    fi

    if [[ "$HOST_TRIPLE" = "$TARGET_TRIPLE" ]]; then
        echo "[TEST] rust-lang/regex tests"
        ../build/cargo-clif test --tests -- --exclude-should-panic --test-threads 1 -Zunstable-options -q
    else
        echo "[AOT] rust-lang/regex tests"
        ../build/cargo-clif build --tests --target $TARGET_TRIPLE
    fi
    popd

    pushd portable-simd
    echo "[TEST] rust-lang/portable-simd"
    ../build/cargo-clif clean
    ../build/cargo-clif build --all-targets --target $TARGET_TRIPLE
    if [[ "$HOST_TRIPLE" = "$TARGET_TRIPLE" ]]; then
        ../build/cargo-clif test -q
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
