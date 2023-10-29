#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/../"

source ./scripts/setup_rust_fork.sh

echo "[TEST] Bootstrap of rustc"
pushd rust
rm -r compiler/rustc_codegen_cranelift/{Cargo.*,src}
cp ../Cargo.* compiler/rustc_codegen_cranelift/
cp -r ../src compiler/rustc_codegen_cranelift/src

# CG_CLIF_FORCE_GNU_AS will force usage of as instead of the LLVM backend of rustc as we
# the LLVM backend isn't compiled in here.
CG_CLIF_FORCE_GNU_AS=1 ./x.py build --stage 1 library/std
popd
