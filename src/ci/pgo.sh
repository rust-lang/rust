#!/bin/bash
# ignore-tidy-linelength

set -euxo pipefail

ci_dir=`cd $(dirname $0) && pwd`
source "$ci_dir/shared.sh"

# The root checkout, where the source is located
CHECKOUT=/checkout

DOWNLOADED_LLVM=/rustroot

# The main directory where the build occurs, which can be different between linux and windows
BUILD_ROOT=$CHECKOUT/obj

if isWindows; then
    CHECKOUT=$(pwd)
    DOWNLOADED_LLVM=$CHECKOUT/citools/clang-rust
    BUILD_ROOT=$CHECKOUT
fi

# The various build artifacts used in other commands: to launch rustc builds, build the perf
# collector, and run benchmarks to gather profiling data
BUILD_ARTIFACTS=$BUILD_ROOT/build/$PGO_HOST
RUSTC_STAGE_0=$BUILD_ARTIFACTS/stage0/bin/rustc
CARGO_STAGE_0=$BUILD_ARTIFACTS/stage0/bin/cargo
RUSTC_STAGE_2=$BUILD_ARTIFACTS/stage2/bin/rustc

# Windows needs these to have the .exe extension
if isWindows; then
    RUSTC_STAGE_0="${RUSTC_STAGE_0}.exe"
    CARGO_STAGE_0="${CARGO_STAGE_0}.exe"
    RUSTC_STAGE_2="${RUSTC_STAGE_2}.exe"
fi

# Make sure we have a temporary PGO work folder
PGO_TMP=/tmp/tmp-pgo
mkdir -p $PGO_TMP
rm -rf $PGO_TMP/*

RUSTC_PERF=$PGO_TMP/rustc-perf

# Compile several crates to gather execution PGO profiles.
# Arg0 => profiles (Debug, Opt)
# Arg1 => scenarios (Full, IncrFull, All)
# Arg2 => crates (syn, cargo, ...)
gather_profiles () {
  cd $BUILD_ROOT

  # Compile libcore, both in opt-level=0 and opt-level=3
  RUSTC_BOOTSTRAP=1 $RUSTC_STAGE_2 \
      --edition=2021 --crate-type=lib $CHECKOUT/library/core/src/lib.rs \
      --out-dir $PGO_TMP
  RUSTC_BOOTSTRAP=1 $RUSTC_STAGE_2 \
      --edition=2021 --crate-type=lib -Copt-level=3 $CHECKOUT/library/core/src/lib.rs \
      --out-dir $PGO_TMP

  cd $RUSTC_PERF

  # Run rustc-perf benchmarks
  # Benchmark using profile_local with eprintln, which essentially just means
  # don't actually benchmark -- just make sure we run rustc a bunch of times.
  RUST_LOG=collector=debug \
  RUSTC=$RUSTC_STAGE_0 \
  RUSTC_BOOTSTRAP=1 \
  $CARGO_STAGE_0 run -p collector --bin collector -- \
      profile_local \
      eprintln \
      $RUSTC_STAGE_2 \
      --id Test \
      --profiles $1 \
      --cargo $CARGO_STAGE_0 \
      --scenarios $2 \
      --include $3

  cd $BUILD_ROOT
}

# This path has to be absolute
LLVM_PROFILE_DIRECTORY_ROOT=$PGO_TMP/llvm-pgo

# We collect LLVM profiling information and rustc profiling information in
# separate phases. This increases build time -- though not by a huge amount --
# but prevents any problems from arising due to different profiling runtimes
# being simultaneously linked in.
# LLVM IR PGO does not respect LLVM_PROFILE_FILE, so we have to set the profiling file
# path through our custom environment variable. We include the PID in the directory path
# to avoid updates to profile files being lost because of race conditions.
LLVM_PROFILE_DIR=${LLVM_PROFILE_DIRECTORY_ROOT}/prof-%p python3 $CHECKOUT/x.py build \
    --target=$PGO_HOST \
    --host=$PGO_HOST \
    --stage 2 library/std \
    --llvm-profile-generate

# Compile rustc-perf:
# - get the expected commit source code: on linux, the Dockerfile downloads a source archive before
# running this script. On Windows, we do that here.
if isLinux; then
    cp -r /tmp/rustc-perf $RUSTC_PERF
    chown -R $(whoami): $RUSTC_PERF
else
    # rustc-perf version from 2022-07-22
    PERF_COMMIT=3c253134664fdcba862c539d37f0de18557a9a4c
    retry curl -LS -o $PGO_TMP/perf.zip \
        https://github.com/rust-lang/rustc-perf/archive/$PERF_COMMIT.zip && \
        cd $PGO_TMP && unzip -q perf.zip && \
        mv rustc-perf-$PERF_COMMIT $RUSTC_PERF && \
        rm perf.zip
fi

# - build rustc-perf's collector ahead of time, which is needed to make sure the rustc-fake binary
# used by the collector is present.
cd $RUSTC_PERF

RUSTC=$RUSTC_STAGE_0 \
RUSTC_BOOTSTRAP=1 \
$CARGO_STAGE_0 build -p collector

# Here we're profiling LLVM, so we only care about `Debug` and `Opt`, because we want to stress
# codegen. We also profile some of the most prolific crates.
gather_profiles "Debug,Opt" "Full" \
    "syn-1.0.89,cargo-0.60.0,serde-1.0.136,ripgrep-13.0.0,regex-1.5.5,clap-3.1.6,hyper-0.14.18"

LLVM_PROFILE_MERGED_FILE=$PGO_TMP/llvm-pgo.profdata

# Merge the profile data we gathered for LLVM
# Note that this uses the profdata from the clang we used to build LLVM,
# which likely has a different version than our in-tree clang.
$DOWNLOADED_LLVM/bin/llvm-profdata merge -o ${LLVM_PROFILE_MERGED_FILE} ${LLVM_PROFILE_DIRECTORY_ROOT}

echo "LLVM PGO statistics"
du -sh ${LLVM_PROFILE_MERGED_FILE}
du -sh ${LLVM_PROFILE_DIRECTORY_ROOT}
echo "Profile file count"
find ${LLVM_PROFILE_DIRECTORY_ROOT} -type f | wc -l

# We don't need the individual .profraw files now that they have been merged into a final .profdata
rm -r $LLVM_PROFILE_DIRECTORY_ROOT

# Rustbuild currently doesn't support rebuilding LLVM when PGO options
# change (or any other llvm-related options); so just clear out the relevant
# directories ourselves.
rm -r $BUILD_ARTIFACTS/llvm $BUILD_ARTIFACTS/lld

# Okay, LLVM profiling is done, switch to rustc PGO.

# The path has to be absolute
RUSTC_PROFILE_DIRECTORY_ROOT=$PGO_TMP/rustc-pgo

python3 $CHECKOUT/x.py build --target=$PGO_HOST --host=$PGO_HOST \
    --stage 2 library/std \
    --rust-profile-generate=${RUSTC_PROFILE_DIRECTORY_ROOT}

# Here we're profiling the `rustc` frontend, so we also include `Check`.
# The benchmark set includes various stress tests that put the frontend under pressure.
if isLinux; then
    # The profile data is written into a single filepath that is being repeatedly merged when each
    # rustc invocation ends. Empirically, this can result in some profiling data being lost. That's
    # why we override the profile path to include the PID. This will produce many more profiling
    # files, but the resulting profile will produce a slightly faster rustc binary.
    LLVM_PROFILE_FILE=${RUSTC_PROFILE_DIRECTORY_ROOT}/default_%m_%p.profraw gather_profiles \
        "Check,Debug,Opt" "All" \
        "externs,ctfe-stress-5,cargo-0.60.0,token-stream-stress,match-stress,tuple-stress,diesel-1.4.8,bitmaps-3.1.0"
else
    # On windows, we don't do that yet (because it generates a lot of data, hitting disk space
    # limits on the builder), and use the default profraw merging behavior.
    gather_profiles \
        "Check,Debug,Opt" "All" \
        "externs,ctfe-stress-5,cargo-0.60.0,token-stream-stress,match-stress,tuple-stress,diesel-1.4.8,bitmaps-3.1.0"
fi

RUSTC_PROFILE_MERGED_FILE=$PGO_TMP/rustc-pgo.profdata

# Merge the profile data we gathered
$BUILD_ARTIFACTS/llvm/bin/llvm-profdata \
    merge -o ${RUSTC_PROFILE_MERGED_FILE} ${RUSTC_PROFILE_DIRECTORY_ROOT}

echo "Rustc PGO statistics"
du -sh ${RUSTC_PROFILE_MERGED_FILE}
du -sh ${RUSTC_PROFILE_DIRECTORY_ROOT}
echo "Profile file count"
find ${RUSTC_PROFILE_DIRECTORY_ROOT} -type f | wc -l

# We don't need the individual .profraw files now that they have been merged into a final .profdata
rm -r $RUSTC_PROFILE_DIRECTORY_ROOT

# Rustbuild currently doesn't support rebuilding LLVM when PGO options
# change (or any other llvm-related options); so just clear out the relevant
# directories ourselves.
rm -r $BUILD_ARTIFACTS/llvm $BUILD_ARTIFACTS/lld

if isLinux; then
  # Gather BOLT profile (BOLT is currently only available on Linux)
  python3 ../x.py build --target=$PGO_HOST --host=$PGO_HOST \
      --stage 2 library/std \
      --llvm-profile-use=${LLVM_PROFILE_MERGED_FILE} \
      --llvm-bolt-profile-generate

  BOLT_PROFILE_MERGED_FILE=/tmp/bolt.profdata

  # Here we're profiling Bolt.
  gather_profiles "Check,Debug,Opt" "Full" \
  "syn-1.0.89,serde-1.0.136,ripgrep-13.0.0,regex-1.5.5,clap-3.1.6,hyper-0.14.18"

  merge-fdata /tmp/prof.fdata* > ${BOLT_PROFILE_MERGED_FILE}

  echo "BOLT statistics"
  du -sh /tmp/prof.fdata*
  du -sh ${BOLT_PROFILE_MERGED_FILE}
  echo "Profile file count"
  find /tmp/prof.fdata* -type f | wc -l

  rm -r $BUILD_ARTIFACTS/llvm $BUILD_ARTIFACTS/lld

  # This produces the actual final set of artifacts, using both the LLVM and rustc
  # collected profiling data.
  $@ \
      --rust-profile-use=${RUSTC_PROFILE_MERGED_FILE} \
      --llvm-profile-use=${LLVM_PROFILE_MERGED_FILE} \
      --llvm-bolt-profile-use=${BOLT_PROFILE_MERGED_FILE}
else
  $@ \
      --rust-profile-use=${RUSTC_PROFILE_MERGED_FILE} \
      --llvm-profile-use=${LLVM_PROFILE_MERGED_FILE}
fi

echo "Rustc binary size"
ls -la ./build/$PGO_HOST/stage2/bin
ls -la ./build/$PGO_HOST/stage2/lib
