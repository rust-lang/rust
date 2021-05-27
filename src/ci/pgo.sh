#!/bin/bash

set -euxo pipefail

rm -rf /tmp/rustc-pgo

python2.7 ../x.py build --target=$PGO_HOST --host=$PGO_HOST \
    --stage 2 library/std --rust-profile-generate=/tmp/rustc-pgo

RUSTC_BOOTSTRAP=1 ./build/$PGO_HOST/stage2/bin/rustc --edition=2018 \
    --crate-type=lib ../library/core/src/lib.rs

# Download and build a single-file stress test benchmark on perf.rust-lang.org.
function pgo_perf_benchmark {
    local PERF=1e19fc4c6168d2f7596e512f42f358f245d8f09d
    local github_prefix=https://raw.githubusercontent.com/rust-lang/rustc-perf/$PERF
    local name=$1
    curl -o /tmp/$name.rs $github_prefix/collector/benchmarks/$name/src/lib.rs

    RUSTC_BOOTSTRAP=1 ./build/$PGO_HOST/stage2/bin/rustc --edition=2018 \
        --crate-type=lib /tmp/$name.rs
}

pgo_perf_benchmark externs
pgo_perf_benchmark ctfe-stress-4

cp -pri ../src/tools/cargo /tmp/cargo

# The Cargo repository does not have a Cargo.lock in it, as it relies on the
# lockfile already present in the rust-lang/rust monorepo. This decision breaks
# down when Cargo is built outside the monorepo though (like in this case),
# resulting in a build without any dependency locking.
#
# To ensure Cargo is built with locked dependencies even during PGO profiling
# the following command copies the monorepo's lockfile into the Cargo temporary
# directory. Cargo will *not* keep that lockfile intact, as it will remove all
# the dependencies Cargo itself doesn't rely on. Still, it will prevent
# building Cargo with arbitrary dependency versions.
#
# See #81378 for the bug that prompted adding this.
cp -p ../Cargo.lock /tmp/cargo

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
