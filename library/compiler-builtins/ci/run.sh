set -ex

cargo=cargo

# Test our implementation
if [ "$XARGO" = "1" ]; then
    # FIXME: currently these tests don't work...
    echo nothing to do
else
    run="cargo test --manifest-path testcrate/Cargo.toml --target $1"
    $run
    $run --release
    $run --features c
    $run --features c --release
    $run --features no-asm
    $run --features no-asm --release
fi

cargo build --target $1
cargo build --target $1 --release
cargo build --target $1 --features c
cargo build --target $1 --release --features c
cargo build --target $1 --features no-asm
cargo build --target $1 --release --features no-asm

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

NM=$(find $(rustc --print sysroot) -name llvm-nm)
if [ "$NM" = "" ]; then
  NM=${PREFIX}nm
fi

if [ -d /target ]; then
    path=/target/${1}/debug/deps/libcompiler_builtins-*.rlib
else
    path=target/${1}/debug/deps/libcompiler_builtins-*.rlib
fi

# Look out for duplicated symbols when we include the compiler-rt (C) implementation
for rlib in $(echo $path); do
    set +x
    echo "================================================================"
    echo checking $rlib for duplicate symbols
    echo "================================================================"

    stdout=$($NM -g --defined-only $rlib 2>&1)
    # NOTE On i586, It's normal that the get_pc_thunk symbol appears several
    # times so ignore it
    #
    # FIXME(#167) - we shouldn't ignore `__builtin_cl` style symbols here.
    set +e
    echo "$stdout" | \
      sort | \
      uniq -d | \
      grep -v __x86.get_pc_thunk | \
      grep -v __builtin_cl | \
      grep -v __builtin_ctz | \
      grep 'T __'

    if test $? = 0; then
        exit 1
    fi

    set -ex
done

rm -f $path

# Verify that we haven't drop any intrinsic/symbol
build_intrinsics="$cargo build --target $1 -v --example intrinsics"
RUSTFLAGS="-C debug-assertions=no" $build_intrinsics
RUSTFLAGS="-C debug-assertions=no" $build_intrinsics --release
RUSTFLAGS="-C debug-assertions=no" $build_intrinsics --features c
RUSTFLAGS="-C debug-assertions=no" $build_intrinsics --features c --release

# Verify that there are no undefined symbols to `panic` within our
# implementations
#
# TODO(#79) fix the undefined references problem for debug-assertions+lto
if [ -z "$DEBUG_LTO_BUILD_DOESNT_WORK" ]; then
  RUSTFLAGS="-C debug-assertions=no" \
    CARGO_INCREMENTAL=0 \
    CARGO_PROFILE_DEV_LTO=true \
    $cargo rustc --features "$INTRINSICS_FEATURES" --target $1 --example intrinsics
fi
CARGO_PROFILE_RELEASE_LTO=true \
  $cargo rustc --features "$INTRINSICS_FEATURES" --target $1 --example intrinsics --release

# Ensure no references to a panicking function
for rlib in $(echo $path); do
    set +ex
    $NM -u $rlib 2>&1 | grep panicking

    if test $? = 0; then
        exit 1
    fi
    set -ex
done

true
