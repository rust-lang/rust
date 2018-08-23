#!/bin/bash

cargo build || exit 1

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
    ar x $1 data.o &&
    chmod +rw data.o &&
    mv data.o $2
    popd
}

RUSTC="rustc -Zcodegen-backend=$(pwd)/target/debug/librustc_codegen_cranelift.$dylib_ext -L crate=target/out --out-dir target/out"

rm -r target/out
mkdir -p target/out/mir
mkdir -p target/out/clif

SHOULD_CODEGEN=1 $RUSTC examples/mini_core.rs --crate-name mini_core --crate-type lib &&
extract_data libmini_core.rlib mini_core.o &&

$RUSTC examples/example.rs --crate-type lib &&

# SimpleJIT is broken
#SHOULD_RUN=1 $RUSTC examples/mini_core_hello_world.rs --crate-type bin &&

$RUSTC examples/mini_core_hello_world.rs --crate-type bin &&
extract_data mini_core_hello_world mini_core_hello_world.o &&

gcc target/out/mini_core.o target/out/mini_core_hello_world.o -o target/out/mini_core_hello_world || exit 1
./target/out/mini_core_hello_world

$RUSTC target/libcore/src/libcore/lib.rs --color=always --crate-type lib -Cincremental=target/incremental 2>&1 | (head -n 20; echo "===="; tail -n 1000)
cat target/out/log.txt | sort | uniq -c
