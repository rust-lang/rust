#!/bin/bash
# This script installs clang on the local machine. Note that we don't install
# clang on Linux since its compiler story is just so different. Each container
# has its own toolchain configured appropriately already.

set -euo pipefail
IFS=$'\n\t'

source "$(cd "$(dirname "$0")" && pwd)/../shared.sh"

if isMacOS; then
    curl -f "${MIRRORS_BASE}/clang%2Bllvm-9.0.0-x86_64-darwin-apple.tar.xz" | tar xJf -

    ciCommandSetEnv CC "$(pwd)/clang+llvm-9.0.0-x86_64-darwin-apple/bin/clang"
    ciCommandSetEnv CXX "$(pwd)/clang+llvm-9.0.0-x86_64-darwin-apple/bin/clang++"

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
    # Original downloaded here came from
    # http://releases.llvm.org/9.0.0/LLVM-9.0.0-win64.exe
    # That installer was run through `wine ./installer.exe /S /NCRC` on Linux
    # and then the resulting installation directory (found in
    # `$HOME/.wine/drive_c/Program Files/LLVM`) was packaged up into a tarball.
    # We've had issues otherwise that the installer will randomly hang, provide
    # not a lot of useful information, pollute global state, etc. In general the
    # tarball is just more confined and easier to deal with when working with
    # various CI environments.

    mkdir -p citools
    cd citools
    curl -f "${MIRRORS_BASE}/LLVM-9.0.0-win64.tar.gz" | tar xzf -
    ciCommandSetEnv RUST_CONFIGURE_ARGS \
        "${RUST_CONFIGURE_ARGS} --set llvm.clang-cl=$(pwd)/clang-rust/bin/clang-cl.exe"
fi
