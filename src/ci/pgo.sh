#!/bin/bash

set -euxo pipefail

rm -rf /tmp/rustc-pgo

python2.7 ../x.py build --target=$PGO_HOST --host=$PGO_HOST \
    --stage 2 library/std --rust-profile-generate=/tmp/rustc-pgo

./build/$PGO_HOST/stage2/bin/rustc --edition=2018 \
    --crate-type=lib ../library/core/src/lib.rs

# Download and build a single-file stress test benchmark on perf.rust-lang.org.
function pgo_perf_benchmark {
    local PERF=e095f5021bf01cf3800f50b3a9f14a9683eb3e4e
    local github_prefix=https://raw.githubusercontent.com/rust-lang/rustc-perf/$PERF
    local name=$1
    curl -o /tmp/$name.rs $github_prefix/collector/benchmarks/$name/src/lib.rs
    ./build/$PGO_HOST/stage2/bin/rustc --edition=2018 --crate-type=lib /tmp/$name.rs
}

pgo_perf_benchmark externs
pgo_perf_benchmark ctfe-stress-4

cp -pri ../src/tools/cargo /tmp/cargo

# Build cargo (with some flags)
function pgo_cargo {
    RUSTC=./build/$PGO_HOST/stage2/bin/rustc \
        ./build/$PGO_HOST/stage0/bin/cargo $@ \
        --manifest-path /tmp/cargo/Cargo.toml
}

# Build a couple different variants of Cargo
CARGO_INCREMENTAL=1 pgo_cargo check
echo 'pub fn barbarbar() {}' >> /tmp/cargo/src/cargo/lib.rs
CARGO_INCREMENTAL=1 pgo_cargo check
touch /tmp/cargo/src/cargo/lib.rs
CARGO_INCREMENTAL=1 pgo_cargo check
pgo_cargo build --release

# Merge the profile data we gathered
./build/$PGO_HOST/llvm/bin/llvm-profdata \
    merge -o /tmp/rustc-pgo.profdata /tmp/rustc-pgo

# This produces the actual final set of artifacts.
$@ --rust-profile-use=/tmp/rustc-pgo.profdata
