#!/bin/bash
set -e

# Settings
export CHANNEL="release"
build_sysroot=1
target_dir='build'
while [[ $# != 0 ]]; do
    case $1 in
        "--debug")
            export CHANNEL="debug"
            ;;
        "--without-sysroot")
            build_sysroot=0
            ;;
        "--target-dir")
            target_dir=$2
            shift
            ;;
        *)
            echo "Unknown flag '$1'"
            echo "Usage: ./build.sh [--debug] [--without-sysroot] [--target-dir DIR]"
            ;;
    esac
    shift
done

# Build cg_clif
unset CARGO_TARGET_DIR
export RUSTFLAGS="-Zrun_dsymutil=no"
unamestr=$(uname)
if [[ "$unamestr" == 'Linux' ]]; then
   export RUSTFLAGS='-Clink-arg=-Wl,-rpath=$ORIGIN/../lib '$RUSTFLAGS
elif [[ "$unamestr" == 'Darwin' ]]; then
   export RUSTFLAGS='-Clink-arg=-Wl,-rpath,@loader_path/../lib -Zosx-rpath-install-name '$RUSTFLAGS
   dylib_ext='dylib'
else
   echo "Unsupported os"
   exit 1
fi
if [[ "$CHANNEL" == "release" ]]; then
    cargo build --release
else
    cargo build
fi

rm -rf "$target_dir"
mkdir "$target_dir"
mkdir "$target_dir"/bin "$target_dir"/lib
ln target/$CHANNEL/cg_clif{,_build_sysroot} "$target_dir"/bin
ln target/$CHANNEL/*rustc_codegen_cranelift* "$target_dir"/lib
ln rust-toolchain scripts/config.sh scripts/cargo.sh "$target_dir"

if [[ "$build_sysroot" == "1" ]]; then
    echo "[BUILD] sysroot"
    export CG_CLIF_INCR_CACHE_DISABLED=1
    dir=$(pwd)
    cd "$target_dir"
    time "$dir/build_sysroot/build_sysroot.sh"
fi
