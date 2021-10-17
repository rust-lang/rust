#!/bin/bash

set -euxo pipefail

rm -rf /tmp/rustc-pgo

# We collect LLVM profiling information and rustc profiling information in
# separate phases. This increases build time -- though not by a huge amount --
# but prevents any problems from arising due to different profiling runtimes
# being simultaneously linked in.

python3 ../x.py build --target=$PGO_HOST --host=$PGO_HOST \
    --stage 2 library/std \
    --llvm-profile-generate

# Profile libcore compilation in opt-level=0 and opt-level=3
RUSTC_BOOTSTRAP=1 ./build/$PGO_HOST/stage2/bin/rustc --edition=2018 \
    --crate-type=lib ../library/core/src/lib.rs
RUSTC_BOOTSTRAP=1 ./build/$PGO_HOST/stage2/bin/rustc --edition=2018 \
    --crate-type=lib -Copt-level=3 ../library/core/src/lib.rs

# Merge the profile data we gathered for LLVM
# Note that this uses the profdata from the clang we used to build LLVM,
# which likely has a different version than our in-tree clang.
/rustroot/bin/llvm-profdata \
    merge -o /tmp/llvm-pgo.profdata ./build/$PGO_HOST/llvm/build/profiles

# Rustbuild currently doesn't support rebuilding LLVM when PGO options
# change (or any other llvm-related options); so just clear out the relevant
# directories ourselves.
rm -r ./build/$PGO_HOST/llvm ./build/$PGO_HOST/lld

# Okay, LLVM profiling is done, switch to rustc PGO.

python3 ../x.py build --target=$PGO_HOST --host=$PGO_HOST \
    --stage 2 library/std \
    --rust-profile-generate=/tmp/rustc-pgo

# Profile libcore compilation in opt-level=0 and opt-level=3
RUSTC_BOOTSTRAP=1 ./build/$PGO_HOST/stage2/bin/rustc --edition=2018 \
    --crate-type=lib ../library/core/src/lib.rs
RUSTC_BOOTSTRAP=1 ./build/$PGO_HOST/stage2/bin/rustc --edition=2018 \
    --crate-type=lib -Copt-level=3 ../library/core/src/lib.rs

cp -r /tmp/rustc-perf ./
chown -R $(whoami): ./rustc-perf
cd rustc-perf

# Build the collector ahead of time, which is needed to make sure the rustc-fake
# binary used by the collector is present.
RUSTC=/checkout/obj/build/$PGO_HOST/stage0/bin/rustc \
RUSTC_BOOTSTRAP=1 \
/checkout/obj/build/$PGO_HOST/stage0/bin/cargo build -p collector

# benchmark using profile_local with eprintln, which essentially just means
# don't actually benchmark -- just make sure we run rustc a bunch of times.
RUST_LOG=collector=debug \
RUSTC=/checkout/obj/build/$PGO_HOST/stage0/bin/rustc \
RUSTC_BOOTSTRAP=1 \
/checkout/obj/build/$PGO_HOST/stage0/bin/cargo run -p collector --bin collector -- \
        profile_local \
        eprintln \
        /checkout/obj/build/$PGO_HOST/stage2/bin/rustc \
        Test \
        --builds Check,Debug,Opt \
        --cargo /checkout/obj/build/$PGO_HOST/stage0/bin/cargo \
        --runs All \
        --include externs,ctfe-stress-4,inflate,cargo,token-stream-stress,match-stress-enum

cd /checkout/obj

# Merge the profile data we gathered
./build/$PGO_HOST/llvm/bin/llvm-profdata \
    merge -o /tmp/rustc-pgo.profdata /tmp/rustc-pgo

# Rustbuild currently doesn't support rebuilding LLVM when PGO options
# change (or any other llvm-related options); so just clear out the relevant
# directories ourselves.
rm -r ./build/$PGO_HOST/llvm ./build/$PGO_HOST/lld

# This produces the actual final set of artifacts, using both the LLVM and rustc
# collected profiling data.
$@ \
    --rust-profile-use=/tmp/rustc-pgo.profdata \
    --llvm-profile-use=/tmp/llvm-pgo.profdata
