set -e

# Test our implementation
case $1 in
    thumb*)
        xargo build --target $1
        xargo build --target $1 --release
        ;;
    *)
        cargo test --no-default-features --target $1
        cargo test --no-default-features --target $1 --release
        ;;
esac

# Verify that we haven't drop any intrinsic/symbol
case $1 in
    thumb*)
        xargo build --features c --target $1 --bin intrinsics
        ;;
    *)
        cargo build --no-default-features --features c --target $1 --bin intrinsics
        ;;
esac

# Verify that there are no undefined symbols to `panic` within our implementations
# TODO(#79) fix the undefined references problem for debug-assertions+lto
case $1 in
    thumb*)
        RUSTFLAGS="-C debug-assertions=no -C link-arg=-nostartfiles" xargo rustc --no-default-features --features c --target $1 --bin intrinsics -- -C lto
        xargo rustc --no-default-features --features c --target $1 --bin intrinsics --release -- -C lto
        ;;
    *)
        RUSTFLAGS="-C debug-assertions=no" cargo rustc --no-default-features --features c --target $1 --bin intrinsics -- -C lto
        cargo rustc --no-default-features --features c --target $1 --bin intrinsics --release -- -C lto
        ;;
esac

# Look out for duplicated symbols when we include the compiler-rt (C) implementation
PREFIX=$(echo $1 | sed -e 's/unknown-//')-
case $1 in
    armv7-*)
        PREFIX=arm-linux-gnueabihf-
        ;;
    thumb*)
        PREFIX=arm-none-eabi-
        ;;
    *86*-*)
        PREFIX=
        ;;
esac

case $TRAVIS_OS_NAME in
    osx)
        # NOTE OSx's nm doesn't accept the `--defined-only` or provide an equivalent.
        # Use GNU nm instead
        NM=gnm
        brew install binutils
        ;;
    *)
        NM=nm
        ;;
esac

# NOTE On i586, It's normal that the get_pc_thunk symbol appears several times so ignore it
if [ $TRAVIS_OS_NAME = osx ]; then
    path=target/${1}/debug/libcompiler_builtins.rlib
else
    path=/target/${1}/debug/libcompiler_builtins.rlib
fi

stdout=$($PREFIX$NM -g --defined-only $path)

set +e
echo "$stdout" | sort | uniq -d | grep -v __x86.get_pc_thunk | grep 'T __'

if test $? = 0; then
    exit 1
fi
