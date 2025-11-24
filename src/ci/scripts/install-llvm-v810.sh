#!/bin/bash
# ignore-tidy-linelength
# Install a set of C compilation tools from the llvm-v810 repo.

set -euo pipefail
IFS=$'\n\t'

source "$(cd "$(dirname "$0")" && pwd)/../shared.sh"

BUILD_DATE="2025-11-22"

if isMacOS; then
    targetName=darwin
    bundleExt=tar.xz
    exeExt=""
elif isWindows; then
    targetName=windows
    bundleExt=7z
    exeExt=".exe"
else
    targetName=linux
    bundleExt=tar.xz
    exeExt=""
fi

curl -o "llvm-v810.$bundleExt" -L "https://github.com/SupernaviX/v810-llvm/releases/download/llvm-v810-$BUILD_DATE/llvm-v810-$targetName-main.$bundleExt"
if [ "$bundleExt" == "7z" ]; then
    7z x "llvm-v810.$bundleExt"
else
    tar -xvf "llvm-v810.$bundleExt"
fi

cp "llvm-v810/bin/llvm-ar$exeExt" "llvm-v810/bin/v810-unknown-vb-llvm-ar$exeExt"
cp "llvm-v810/bin/clang$exeExt" "llvm-v810/bin/v810-unknown-vb-clang$exeExt"
cp "llvm-v810/bin/clang++$exeExt" "llvm-v810/bin/v810-unknown-vb-clang++$exeExt"
cp "llvm-v810/bin/llvm-ranlib$exeExt" "llvm-v810/bin/v810-unknown-vb-ranlib$exeExt"
cp "llvm-v810/bin/ld.lld$exeExt" "llvm-v810/bin/v810-unknown-vb-ld.lld$exeExt"
ciCommandAddPath "$(pwd)/llvm-v810/bin"

if isLinux; then
    sudo apt-get install cmake
fi
