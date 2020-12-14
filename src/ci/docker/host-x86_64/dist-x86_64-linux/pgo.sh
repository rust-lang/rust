#!/bin/bash

set -euxo pipefail

rm -rf /tmp/rustc-pgo

python2.7 ../x.py build --stage 2 library/std --rust-profile-generate=/tmp/rustc-pgo

./build/x86_64-unknown-linux-gnu/stage2/bin/rustc --edition=2018 \
    --crate-type=lib ../library/core/src/lib.rs

PERF=e095f5021bf01cf3800f50b3a9f14a9683eb3e4e

curl -o /tmp/externs.rs \
https://raw.githubusercontent.com/rust-lang/rustc-perf/$PERF/collector/benchmarks/externs/src/lib.rs
./build/x86_64-unknown-linux-gnu/stage2/bin/rustc --edition=2018 --crate-type=lib /tmp/externs.rs

curl -o /tmp/ctfe.rs \
https://raw.githubusercontent.com/rust-lang/rustc-perf/$PERF/collector/benchmarks/ctfe-stress-4/src/lib.rs
./build/x86_64-unknown-linux-gnu/stage2/bin/rustc --edition=2018 --crate-type=lib /tmp/ctfe.rs

cp -pri ../src/tools/cargo /tmp/cargo

RUSTC=./build/x86_64-unknown-linux-gnu/stage2/bin/rustc CARGO_INCREMENTAL=1 \
    ./build/x86_64-unknown-linux-gnu/stage0/bin/cargo check \
    --manifest-path /tmp/cargo/Cargo.toml
echo 'pub fn barbarbar() {}' >> /tmp/cargo/src/cargo/lib.rs
RUSTC=./build/x86_64-unknown-linux-gnu/stage2/bin/rustc CARGO_INCREMENTAL=1 \
    ./build/x86_64-unknown-linux-gnu/stage0/bin/cargo check \
    --manifest-path /tmp/cargo/Cargo.toml
touch /tmp/cargo/src/cargo/lib.rs
RUSTC=./build/x86_64-unknown-linux-gnu/stage2/bin/rustc CARGO_INCREMENTAL=1 \
    ./build/x86_64-unknown-linux-gnu/stage0/bin/cargo check \
    --manifest-path /tmp/cargo/Cargo.toml
RUSTC=./build/x86_64-unknown-linux-gnu/stage2/bin/rustc CARGO_INCREMENTAL=1 \
    ./build/x86_64-unknown-linux-gnu/stage0/bin/cargo check \
    --manifest-path /tmp/cargo/Cargo.toml
RUSTC=./build/x86_64-unknown-linux-gnu/stage2/bin/rustc \
    ./build/x86_64-unknown-linux-gnu/stage0/bin/cargo build --release \
    --manifest-path /tmp/cargo/Cargo.toml
./build/x86_64-unknown-linux-gnu/llvm/bin/llvm-profdata \
    merge -o /tmp/rustc-pgo.profdata /tmp/rustc-pgo

cp /tmp/rustc-pgo.profdata ./build/dist/rustc-pgo.profdata

# This produces the actual final set of artifacts
python2.7 ../x.py dist --rust-profile-use=/tmp/rustc-pgo.profdata \
    --host $HOSTS --target $HOSTS \
    --include-default-paths \
    src/tools/build-manifest
