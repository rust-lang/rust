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

if [[ "$1" == "--release" ]]; then
    channel='release'
    cargo build --release
else
    channel='debug'
    cargo build
fi

export RUSTFLAGS='-Zalways-encode-mir -Cpanic=abort -Cdebuginfo=2 -Zcodegen-backend='$(pwd)'/target/'$channel'/librustc_codegen_cranelift.'$dylib_ext' --sysroot '$(pwd)'/build_sysroot/sysroot'
RUSTC="rustc $RUSTFLAGS -L crate=target/out --out-dir target/out"
export RUSTC_LOG=warn # display metadata load errors
