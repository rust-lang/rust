set -e

export CARGO_INCREMENTAL=0

if [ -f ./gcc_path ]; then 
    export GCC_PATH=$(cat gcc_path)
else
    echo 'Please put the path to your custom build of libgccjit in the file `gcc_path`, see Readme.md for details'
    exit 1
fi

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
#TARGET_TRIPLE="m68k-unknown-linux-gnu"

linker=''
RUN_WRAPPER=''
if [[ "$HOST_TRIPLE" != "$TARGET_TRIPLE" ]]; then
   if [[ "$TARGET_TRIPLE" == "m68k-unknown-linux-gnu" ]]; then
       TARGET_TRIPLE="mips-unknown-linux-gnu"
       linker='-Clinker=m68k-linux-gcc'
   elif [[ "$TARGET_TRIPLE" == "aarch64-unknown-linux-gnu" ]]; then
      # We are cross-compiling for aarch64. Use the correct linker and run tests in qemu.
      linker='-Clinker=aarch64-linux-gnu-gcc'
      RUN_WRAPPER='qemu-aarch64 -L /usr/aarch64-linux-gnu'
   else
      echo "Unknown non-native platform"
   fi
fi

export RUSTFLAGS="$linker -Cpanic=abort -Csymbol-mangling-version=v0 -Cdebuginfo=2 -Clto=off -Zpanic-abort-tests -Zcodegen-backend=$(pwd)/target/${CHANNEL:-debug}/librustc_codegen_gcc.$dylib_ext --sysroot $(pwd)/build_sysroot/sysroot"

# FIXME(antoyo): remove once the atomic shim is gone
if [[ `uname` == 'Darwin' ]]; then
   export RUSTFLAGS="$RUSTFLAGS -Clink-arg=-undefined -Clink-arg=dynamic_lookup"
fi

RUSTC="rustc $RUSTFLAGS -L crate=target/out --out-dir target/out"
export RUSTC_LOG=warn # display metadata load errors

export LD_LIBRARY_PATH="$(pwd)/target/out:$(pwd)/build_sysroot/sysroot/lib/rustlib/$TARGET_TRIPLE/lib:$GCC_PATH"
export DYLD_LIBRARY_PATH=$LD_LIBRARY_PATH
