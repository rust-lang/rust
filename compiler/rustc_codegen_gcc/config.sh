set -e

export CARGO_INCREMENTAL=0

if [ -f ./gcc_path ]; then
    export GCC_PATH=$(cat gcc_path)
elif (( $use_system_gcc == 1 )); then
    echo 'Using system GCC'
else
    echo 'Please put the path to your custom build of libgccjit in the file `gcc_path`, see Readme.md for details'
    exit 1
fi

if [[ -z "$RUSTC" ]]; then
    export RUSTC="rustc"
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

HOST_TRIPLE=$($RUSTC -vV | grep host | cut -d: -f2 | tr -d " ")
# TODO: remove $OVERWRITE_TARGET_TRIPLE when config.sh is removed.
TARGET_TRIPLE="${OVERWRITE_TARGET_TRIPLE:-$HOST_TRIPLE}"

linker=''
RUN_WRAPPER=''
if [[ "$HOST_TRIPLE" != "$TARGET_TRIPLE" ]]; then
    RUN_WRAPPER=run_in_vm
    if [[ "$TARGET_TRIPLE" == "m68k-unknown-linux-gnu" ]]; then
        linker='-Clinker=m68k-unknown-linux-gnu-gcc'
    elif [[ "$TARGET_TRIPLE" == "aarch64-unknown-linux-gnu" ]]; then
        # We are cross-compiling for aarch64. Use the correct linker and run tests in qemu.
        linker='-Clinker=aarch64-linux-gnu-gcc'
    else
        echo "Unknown non-native platform"
    fi
fi

# Since we don't support ThinLTO, disable LTO completely when not trying to do LTO.
# TODO(antoyo): remove when we can handle ThinLTO.
disable_lto_flags=''
if [[ ! -v FAT_LTO ]]; then
    disable_lto_flags='-Clto=off'
fi

if [[ -z "$BUILTIN_BACKEND" ]]; then
    export RUSTFLAGS="$CG_RUSTFLAGS $linker -Csymbol-mangling-version=v0 -Cdebuginfo=2 $disable_lto_flags -Zcodegen-backend=$(pwd)/target/${CHANNEL:-debug}/librustc_codegen_gcc.$dylib_ext --sysroot $(pwd)/build_sysroot/sysroot $TEST_FLAGS"
else
    export RUSTFLAGS="$CG_RUSTFLAGS $linker -Csymbol-mangling-version=v0 -Cdebuginfo=2 $disable_lto_flags -Zcodegen-backend=gcc $TEST_FLAGS -Cpanic=abort"

    if [[ ! -z "$RUSTC_SYSROOT" ]]; then
        export RUSTFLAGS="$RUSTFLAGS --sysroot $RUSTC_SYSROOT"
    fi
fi

# FIXME(antoyo): remove once the atomic shim is gone
if [[ unamestr == 'Darwin' ]]; then
    export RUSTFLAGS="$RUSTFLAGS -Clink-arg=-undefined -Clink-arg=dynamic_lookup"
fi

if [[ -z "$cargo_target_dir" ]]; then
    RUST_CMD="$RUSTC $RUSTFLAGS -L crate=target/out --out-dir target/out"
    cargo_target_dir="target/out"
else
    RUST_CMD="$RUSTC $RUSTFLAGS -L crate=$cargo_target_dir --out-dir $cargo_target_dir"
fi
export RUSTC_LOG=warn # display metadata load errors

export LD_LIBRARY_PATH="$(pwd)/target/out:$(pwd)/build_sysroot/sysroot/lib/rustlib/$TARGET_TRIPLE/lib"
if [[ ! -z "$:$GCC_PATH" ]]; then
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$GCC_PATH"
fi

export DYLD_LIBRARY_PATH=$LD_LIBRARY_PATH
# NOTE: To avoid the -fno-inline errors, use /opt/gcc/bin/gcc instead of cc.
# To do so, add a symlink for cc to /opt/gcc/bin/gcc in our PATH.
# Another option would be to add the following Rust flag: -Clinker=/opt/gcc/bin/gcc
export PATH="/opt/gcc/bin:/opt/m68k-unknown-linux-gnu/bin:$PATH"
