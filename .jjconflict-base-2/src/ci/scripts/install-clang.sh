#!/bin/bash
# ignore-tidy-linelength
# This script installs clang on the local machine. Note that we don't install
# clang on Linux since its compiler story is just so different. Each container
# has its own toolchain configured appropriately already.

set -euo pipefail
IFS=$'\n\t'

source "$(cd "$(dirname "$0")" && pwd)/../shared.sh"

# Update Windows's tarballs when bumping the version here.
# Try to keep this in sync with src/ci/docker/scripts/build-clang.sh
LLVM_VERSION="20.1.3"

if isWindows && ! isKnownToBeMingwBuild; then
    # If we're compiling for MSVC then we, like most other distribution builders,
    # switch to clang as the compiler. This'll allow us eventually to enable LTO
    # amongst LLVM and rustc. Note that we only do this on MSVC as I don't think
    # clang has an output mode compatible with MinGW that we need. If it does we
    # should switch to clang for MinGW as well!
    #
    # The LLVM installer is an NSIS installer, which we can extract with 7z. We
    # don't want to run the installer directly; extracting it is more reliable
    # in CI environments.

    mkdir -p citools/clang-rust
    cd citools

    if [[ "${CI_JOB_NAME}" = *aarch64* ]]; then
        suffix=woa64

        # On Arm64, the Ring crate requires that Clang be on the PATH.
        # https://github.com/briansmith/ring/blob/main/BUILDING.md
        ciCommandAddPath "$(cygpath -m "$(pwd)/clang-rust/bin")"
    else
        suffix=win64
    fi
    retry curl -f "${MIRRORS_BASE}/LLVM-${LLVM_VERSION}-${suffix}.exe" \
        -o "LLVM-${LLVM_VERSION}-${suffix}.exe"
    7z x -oclang-rust/ "LLVM-${LLVM_VERSION}-${suffix}.exe"
    ciCommandSetEnv RUST_CONFIGURE_ARGS \
        "${RUST_CONFIGURE_ARGS} --set llvm.clang-cl=$(pwd)/clang-rust/bin/clang-cl.exe"

    # Disable downloading CI LLVM on this builder;
    # setting up clang-cl just above conflicts with the default if-unchanged option.
    ciCommandSetEnv NO_DOWNLOAD_CI_LLVM 1
fi

if isWindows; then
    # GitHub image 20210928.2 added LLVM, but it is broken (and we don't want
    # to use it anyways).
    rm -rf /c/Program\ Files/LLVM
fi
