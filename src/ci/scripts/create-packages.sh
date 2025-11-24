#!/bin/bash
# Run x.py to actually package this

set -euo pipefail
IFS=$'\n\t'

source "$(cd "$(dirname "$0")" && pwd)/../shared.sh"

if isMacOS; then
    nativeTriple=aarch64-apple-darwin
    filename=darwin
elif isWindows; then
    nativeTriple=x86_64-pc-windows-msvc
    filename=windows
else
    nativeTriple=x86_64-unknown-linux-gnu
    filename=linux
fi

cp bootstrap.ci.toml bootstrap.toml
./x.py dist --target "$nativeTriple,v810-unknown-vb"

ls -la build/dist

cp build/dist/rust-nightly-$nativeTriple.tar.xz rust-v810-$filename.tar.xz

if isWindows; then
    cp build/dist/rust-nightly-$nativeTriple.msi rust-v810-$filename.msi
    cp build/dist/rust-src-nightly.tar.xz rust-src-v810.tar.xz
fi
