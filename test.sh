#!/bin/bash

# TODO(antoyo): rewrite to cargo-make (or just) or something like that to only rebuild the sysroot when needed?

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
    shift
else
    echo $LD_LIBRARY_PATH
    export CHANNEL='debug'
    cargo rustc
fi

source config.sh

function clean() {
    rm -r target/out || true
    mkdir -p target/out/gccjit
}

function mini_tests() {
    echo "[BUILD] mini_core"
    $RUSTC example/mini_core.rs --crate-name mini_core --crate-type lib,dylib --target $TARGET_TRIPLE

    echo "[BUILD] example"
    $RUSTC example/example.rs --crate-type lib --target $TARGET_TRIPLE

    echo "[AOT] mini_core_hello_world"
    $RUSTC example/mini_core_hello_world.rs --crate-name mini_core_hello_world --crate-type bin -g --target $TARGET_TRIPLE
    $RUN_WRAPPER ./target/out/mini_core_hello_world abc bcd
}

function build_sysroot() {
    echo "[BUILD] sysroot"
    time ./build_sysroot/build_sysroot.sh
}

function std_tests() {
    echo "[AOT] arbitrary_self_types_pointers_and_wrappers"
    $RUSTC example/arbitrary_self_types_pointers_and_wrappers.rs --crate-name arbitrary_self_types_pointers_and_wrappers --crate-type bin --target $TARGET_TRIPLE
    $RUN_WRAPPER ./target/out/arbitrary_self_types_pointers_and_wrappers

    echo "[AOT] alloc_system"
    $RUSTC example/alloc_system.rs --crate-type lib --target "$TARGET_TRIPLE"

    echo "[AOT] alloc_example"
    $RUSTC example/alloc_example.rs --crate-type bin --target $TARGET_TRIPLE
    $RUN_WRAPPER ./target/out/alloc_example

    echo "[AOT] dst_field_align"
    # FIXME(antoyo): Re-add -Zmir-opt-level=2 once rust-lang/rust#67529 is fixed.
    $RUSTC example/dst-field-align.rs --crate-name dst_field_align --crate-type bin --target $TARGET_TRIPLE
    $RUN_WRAPPER ./target/out/dst_field_align || (echo $?; false)

    echo "[AOT] std_example"
    $RUSTC example/std_example.rs --crate-type bin --target $TARGET_TRIPLE
    $RUN_WRAPPER ./target/out/std_example --target $TARGET_TRIPLE

    echo "[AOT] subslice-patterns-const-eval"
    $RUSTC example/subslice-patterns-const-eval.rs --crate-type bin -Cpanic=abort --target $TARGET_TRIPLE
    $RUN_WRAPPER ./target/out/subslice-patterns-const-eval

    echo "[AOT] track-caller-attribute"
    $RUSTC example/track-caller-attribute.rs --crate-type bin -Cpanic=abort --target $TARGET_TRIPLE
    $RUN_WRAPPER ./target/out/track-caller-attribute

    echo "[BUILD] mod_bench"
    $RUSTC example/mod_bench.rs --crate-type bin --target $TARGET_TRIPLE
}

# FIXME(antoyo): linker gives multiple definitions error on Linux
#echo "[BUILD] sysroot in release mode"
#./build_sysroot/build_sysroot.sh --release

# TODO(antoyo): uncomment when it works.
#pushd simple-raytracer
#if [[ "$HOST_TRIPLE" = "$TARGET_TRIPLE" ]]; then
    #echo "[BENCH COMPILE] ebobby/simple-raytracer"
    #hyperfine --runs ${RUN_RUNS:-10} --warmup 1 --prepare "rm -r target/*/debug || true" \
    #"RUSTFLAGS='' cargo build --target $TARGET_TRIPLE" \
    #"../cargo.sh build"

    #echo "[BENCH RUN] ebobby/simple-raytracer"
    #cp ./target/*/debug/main ./raytracer_cg_gccjit
    #hyperfine --runs ${RUN_RUNS:-10} ./raytracer_cg_llvm ./raytracer_cg_gccjit
#else
    #echo "[BENCH COMPILE] ebobby/simple-raytracer (skipped)"
    #echo "[COMPILE] ebobby/simple-raytracer"
    #../cargo.sh build
    #echo "[BENCH RUN] ebobby/simple-raytracer (skipped)"
#fi
#popd

function test_libcore() {
    pushd build_sysroot/sysroot_src/library/core/tests
    echo "[TEST] libcore"
    rm -r ./target || true
    ../../../../../cargo.sh test
    popd
}

# TODO(antoyo): uncomment when it works.
#pushd regex
#echo "[TEST] rust-lang/regex example shootout-regex-dna"
#../cargo.sh clean
## Make sure `[codegen mono items] start` doesn't poison the diff
#../cargo.sh build --example shootout-regex-dna
#cat examples/regexdna-input.txt | ../cargo.sh run --example shootout-regex-dna | grep -v "Spawned thread" > res.txt
#diff -u res.txt examples/regexdna-output.txt

#echo "[TEST] rust-lang/regex tests"
#../cargo.sh test --tests -- --exclude-should-panic --test-threads 1 -Zunstable-options
#popd

#echo
#echo "[BENCH COMPILE] mod_bench"

#COMPILE_MOD_BENCH_INLINE="$RUSTC example/mod_bench.rs --crate-type bin -Zmir-opt-level=3 -O --crate-name mod_bench_inline"
#COMPILE_MOD_BENCH_LLVM_0="rustc example/mod_bench.rs --crate-type bin -Copt-level=0 -o target/out/mod_bench_llvm_0 -Cpanic=abort"
#COMPILE_MOD_BENCH_LLVM_1="rustc example/mod_bench.rs --crate-type bin -Copt-level=1 -o target/out/mod_bench_llvm_1 -Cpanic=abort"
#COMPILE_MOD_BENCH_LLVM_2="rustc example/mod_bench.rs --crate-type bin -Copt-level=2 -o target/out/mod_bench_llvm_2 -Cpanic=abort"
#COMPILE_MOD_BENCH_LLVM_3="rustc example/mod_bench.rs --crate-type bin -Copt-level=3 -o target/out/mod_bench_llvm_3 -Cpanic=abort"

## Use 100 runs, because a single compilations doesn't take more than ~150ms, so it isn't very slow
#hyperfine --runs ${COMPILE_RUNS:-100} "$COMPILE_MOD_BENCH_INLINE" "$COMPILE_MOD_BENCH_LLVM_0" "$COMPILE_MOD_BENCH_LLVM_1" "$COMPILE_MOD_BENCH_LLVM_2" "$COMPILE_MOD_BENCH_LLVM_3"

#echo
#echo "[BENCH RUN] mod_bench"
#hyperfine --runs ${RUN_RUNS:-10} ./target/out/mod_bench{,_inline} ./target/out/mod_bench_llvm_*

function test_rustc() {
    echo
    echo "[TEST] rust-lang/rust"

    rust_toolchain=$(cat rust-toolchain)

    git clone https://github.com/rust-lang/rust.git || true
    cd rust
    git fetch
    git checkout $(rustc -V | cut -d' ' -f3 | tr -d '(')
    export RUSTFLAGS=

    rm config.toml || true

    cat > config.toml <<EOF
[rust]
codegen-backends = []
deny-warnings = false

[build]
cargo = "$(which cargo)"
local-rebuild = true
rustc = "$HOME/.rustup/toolchains/$rust_toolchain-$TARGET_TRIPLE/bin/rustc"
EOF

    rustc -V | cut -d' ' -f3 | tr -d '('
    git checkout $(rustc -V | cut -d' ' -f3 | tr -d '(') src/test

    for test in $(rg -i --files-with-matches "//(\[\w+\])?~|// error-pattern:|// build-fail|// run-fail|-Cllvm-args" src/test/ui); do
      rm $test
    done

    git checkout -- src/test/ui/issues/auxiliary/issue-3136-a.rs # contains //~ERROR, but shouldn't be removed

    rm -r src/test/ui/{abi*,extern/,llvm-asm/,panic-runtime/,panics/,unsized-locals/,proc-macro/,threads-sendsync/,thinlto/,simd*,borrowck/,test*,*lto*.rs} || true
    for test in $(rg --files-with-matches "catch_unwind|should_panic|thread|lto" src/test/ui); do
      rm $test
    done
    git checkout src/test/ui/type-alias-impl-trait/auxiliary/cross_crate_ice.rs
    git checkout src/test/ui/type-alias-impl-trait/auxiliary/cross_crate_ice2.rs
    rm src/test/ui/llvm-asm/llvm-asm-in-out-operand.rs || true # TODO(antoyo): Enable back this test if I ever implement the llvm_asm! macro.

    RUSTC_ARGS="-Zpanic-abort-tests -Zsymbol-mangling-version=v0 -Zcodegen-backend="$(pwd)"/../target/"$CHANNEL"/librustc_codegen_gcc."$dylib_ext" --sysroot "$(pwd)"/../build_sysroot/sysroot -Cpanic=abort"

    echo "[TEST] rustc test suite"
    COMPILETEST_FORCE_STAGE0=1 ./x.py test --run always --stage 0 src/test/ui/ --rustc-args "$RUSTC_ARGS"
}

function clean_ui_tests() {
    find rust/build/x86_64-unknown-linux-gnu/test/ui/ -name stamp -exec rm -rf {} \;
}

case $1 in
    "--test-rustc")
        test_rustc
        ;;

    "--test-libcore")
        test_libcore
        ;;

    "--clean-ui-tests")
        clean_ui_tests
        ;;

    *)
        clean
        mini_tests
        build_sysroot
        std_tests
        test_libcore
        test_rustc
        ;;
esac
