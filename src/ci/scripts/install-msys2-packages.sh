#!/bin/bash

set -euo pipefail
IFS=$'\n\t'

source "$(cd "$(dirname "$0")" && pwd)/../shared.sh"

if isWindows; then
    # FIXME(mati865): temporary workaround until chocolatey updates their MSYS2
    base_url='https://ci-mirrors.rust-lang.org/rustc/msys2-repo/msys/x86_64'
    curl ${base_url}/libzstd-1.4.4-2-x86_64.pkg.tar.xz -o libzstd-1.4.4-2-x86_64.pkg.tar.xz
    curl ${base_url}/pacman-5.2.1-6-x86_64.pkg.tar.xz -o pacman-5.2.1-6-x86_64.pkg.tar.xz
    curl ${base_url}/zstd-1.4.4-2-x86_64.pkg.tar.xz -o zstd-1.4.4-2-x86_64.pkg.tar.xz
    pacman -U --noconfirm libzstd-1.4.4-2-x86_64.pkg.tar.xz pacman-5.2.1-6-x86_64.pkg.tar.xz \
        zstd-1.4.4-2-x86_64.pkg.tar.xz
    rm libzstd-1.4.4-2-x86_64.pkg.tar.xz pacman-5.2.1-6-x86_64.pkg.tar.xz \
        zstd-1.4.4-2-x86_64.pkg.tar.xz
    pacman -Sy

    pacman -S --noconfirm --needed base-devel ca-certificates make diffutils tar \
        binutils

    # Detect the native Python version installed on the agent. On GitHub
    # Actions, the C:\hostedtoolcache\windows\Python directory contains a
    # subdirectory for each installed Python version.
    #
    # The -V flag of the sort command sorts the input by version number.
    native_python_version="$(ls /c/hostedtoolcache/windows/Python | sort -Vr | head -n 1)"

    # Make sure we use the native python interpreter instead of some msys equivalent
    # one way or another. The msys interpreters seem to have weird path conversions
    # baked in which break LLVM's build system one way or another, so let's use the
    # native version which keeps everything as native as possible.
    python_home="/c/hostedtoolcache/windows/Python/${native_python_version}/x64"
    cp "${python_home}/python.exe" "${python_home}/python3.exe"
    ciCommandAddPath "C:\\hostedtoolcache\\windows\\Python\\${native_python_version}\\x64"
    ciCommandAddPath "C:\\hostedtoolcache\\windows\\Python\\${native_python_version}\\x64\\Scripts"
fi
