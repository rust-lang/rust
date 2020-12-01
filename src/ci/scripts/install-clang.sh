#!/bin/bash
# This script installs clang on the local machine. Note that we don't install
# clang on Linux since its compiler story is just so different. Each container
# has its own toolchain configured appropriately already.

set -euo pipefail
IFS=$'\n\t'

source "$(cd "$(dirname "$0")" && pwd)/../shared.sh"

# Update both macOS's and Windows's tarballs when bumping the version here.
LLVM_VERSION="10.0.0"

if isMacOS; then
    # If the job selects a specific Xcode version, use that instead of
    # downloading our own version.
    if [[ ${USE_XCODE_CLANG-0} -eq 1 ]]; then
        bindir="$(xcode-select --print-path)/Toolchains/XcodeDefault.xctoolchain/usr/bin"
    else
        file="${MIRRORS_BASE}/clang%2Bllvm-${LLVM_VERSION}-x86_64-apple-darwin.tar.xz"
        curl -f "${file}" | tar xJf -
        bindir="$(pwd)/clang+llvm-${LLVM_VERSION}-x86_64-apple-darwin/bin"
    fi

    ciCommandSetEnv CC "${bindir}/clang"
    ciCommandSetEnv CXX "${bindir}/clang++"

    # macOS 10.15 onwards doesn't have libraries in /usr/include anymore: those
    # are now located deep into the filesystem, under Xcode's own files. The
    # native clang is configured to use the correct path, but our custom one
    # doesn't. This sets the SDKROOT environment variable to the SDK so that
    # our own clang can figure out the correct include path on its own.
    ciCommandSetEnv SDKROOT "$(xcrun --sdk macosx --show-sdk-path)"

    # Configure `AR` specifically so rustbuild doesn't try to infer it as
    # `clang-ar` by accident.
    ciCommandSetEnv AR "ar"
elif isWindows && [[ ${CUSTOM_MINGW-0} -ne 1 ]]; then
    # If we're compiling for MSVC then we, like most other distribution builders,
    # switch to clang as the compiler. This'll allow us eventually to enable LTO
    # amongst LLVM and rustc. Note that we only do this on MSVC as I don't think
    # clang has an output mode compatible with MinGW that we need. If it does we
    # should switch to clang for MinGW as well!
    #
    # Note that the LLVM installer is an NSIS installer
    #
    # Original downloaded here came from:
    #
    #   https://github.com/llvm/llvm-project/releases/download/llvmorg-10.0.0/LLVM-10.0.0-win64.exe
    #
    # That installer was run through `wine ./installer.exe /S /NCRC` on Linux
    # and then the resulting installation directory (found in
    # `$HOME/.wine/drive_c/Program Files/LLVM`) was packaged up into a tarball.
    # We've had issues otherwise that the installer will randomly hang, provide
    # not a lot of useful information, pollute global state, etc. In general the
    # tarball is just more confined and easier to deal with when working with
    # various CI environments.

    mkdir -p citools
    cd citools
    curl -f "${MIRRORS_BASE}/LLVM-${LLVM_VERSION}-win64.tar.gz" | tar xzf -
    ciCommandSetEnv RUST_CONFIGURE_ARGS \
        "${RUST_CONFIGURE_ARGS} --set llvm.clang-cl=$(pwd)/clang-rust/bin/clang-cl.exe"
fi
