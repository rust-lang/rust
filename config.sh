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

export RUSTFLAGS='-Zalways-encode-mir -Cpanic=abort -Zcodegen-backend='$(pwd)'/target/'$channel'/librustc_codegen_cranelift.'$dylib_ext
export XARGO_RUST_SRC=$(pwd)'/target/libcore/src'
RUSTC="rustc $RUSTFLAGS -L crate=target/out --out-dir target/out"
