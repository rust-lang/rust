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

TARGET_TRIPLE=$(rustc -vV | grep host | cut -d: -f2 | tr -d " ")

export RUSTFLAGS='-Zalways-encode-mir -Cpanic=abort -Cdebuginfo=2 -Zcodegen-backend='$(pwd)'/target/'$CHANNEL'/librustc_codegen_cranelift.'$dylib_ext' --sysroot '$(pwd)'/build_sysroot/sysroot'
RUSTC="rustc $RUSTFLAGS -L crate=target/out --out-dir target/out"
export RUSTC_LOG=warn # display metadata load errors

export LD_LIBRARY_PATH=$(pwd)/target/out
export DYLD_LIBRARY_PATH=$(pwd)/target/out
