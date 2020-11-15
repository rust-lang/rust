#!/usr/bin/env bash
set -e

unamestr=`uname`
if [[ "$unamestr" == 'Linux' ]]; then
   dylib_ext='so'
elif [[ "$unamestr" == 'Darwin' ]]; then
   dylib_ext='dylib'
else
   echo "Unsupported os"
   exit 1
fi

HOST_TRIPLE=$(rustc -vV | grep host | cut -d: -f2 | tr -d " ")
TARGET_TRIPLE=$HOST_TRIPLE
#TARGET_TRIPLE="x86_64-pc-windows-gnu"
#TARGET_TRIPLE="aarch64-unknown-linux-gnu"

linker=''
RUN_WRAPPER=''
export JIT_SUPPORTED=1
if [[ "$HOST_TRIPLE" != "$TARGET_TRIPLE" ]]; then
   export JIT_SUPPORTED=0
   if [[ "$TARGET_TRIPLE" == "aarch64-unknown-linux-gnu" ]]; then
      # We are cross-compiling for aarch64. Use the correct linker and run tests in qemu.
      linker='-Clinker=aarch64-linux-gnu-gcc'
      RUN_WRAPPER='qemu-aarch64 -L /usr/aarch64-linux-gnu'
   elif [[ "$TARGET_TRIPLE" == "x86_64-pc-windows-gnu" ]]; then
      # We are cross-compiling for Windows. Run tests in wine.
      RUN_WRAPPER='wine'
   else
      echo "Unknown non-native platform"
   fi
fi

if echo "$RUSTC_WRAPPER" | grep sccache; then
echo
echo -e "\x1b[1;93m=== Warning: Unset RUSTC_WRAPPER to prevent interference with sccache ===\x1b[0m"
echo
export RUSTC_WRAPPER=
fi

dir=$(cd $(dirname "$BASH_SOURCE"); pwd)

export RUSTC=$dir"/cg_clif"
export RUSTFLAGS=$linker
export RUSTDOCFLAGS=$linker' -Ztrim-diagnostic-paths=no -Cpanic=abort -Zpanic-abort-tests '\
'-Zcodegen-backend='$dir'/librustc_codegen_cranelift.'$dylib_ext' --sysroot '$dir'/sysroot'

# FIXME remove once the atomic shim is gone
if [[ `uname` == 'Darwin' ]]; then
   export RUSTFLAGS="$RUSTFLAGS -Clink-arg=-undefined -Clink-arg=dynamic_lookup"
fi

export LD_LIBRARY_PATH="$dir:$(rustc --print sysroot)/lib:$dir/target/out:$dir/sysroot/lib/rustlib/"$TARGET_TRIPLE"/lib"
export DYLD_LIBRARY_PATH=$LD_LIBRARY_PATH

export CG_CLIF_DISPLAY_CG_TIME=1
