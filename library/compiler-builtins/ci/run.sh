set -ex

# Test our implementation
if [ "$NO_STD" = "1" ]; then
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

if [ -d /target ]; then
    path=/target/${1}/debug/deps/libcompiler_builtins-*.rlib
else
    path=target/${1}/debug/deps/libcompiler_builtins-*.rlib
fi

# Remove any existing artifacts from previous tests that don't set #![compiler_builtins]
rm -f $path

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

NM=$(find $(rustc --print sysroot) \( -name llvm-nm -o -name llvm-nm.exe \) )
if [ "$NM" = "" ]; then
  NM=${PREFIX}nm
fi
# i686-pc-windows-gnu tools have a dependency on some DLLs, so run it with
# rustup run to ensure that those are in PATH.
TOOLCHAIN=$(rustup show active-toolchain | sed 's/ (default)//')
if [[ $TOOLCHAIN == *i686-pc-windows-gnu ]]; then
  NM="rustup run $TOOLCHAIN $NM"
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
      grep -v __builtin_sadd_overflow | \
      grep 'T __'

    if test $? = 0; then
        exit 1
    fi

    set -ex
done

rm -f $path

# Verify that we haven't drop any intrinsic/symbol
build_intrinsics="cargo build --target $1 -v --example intrinsics"
$build_intrinsics
$build_intrinsics --release
$build_intrinsics --features c
$build_intrinsics --features c --release

# Verify that there are no undefined symbols to `panic` within our
# implementations
CARGO_PROFILE_DEV_LTO=true \
    cargo build --target $1 --example intrinsics
CARGO_PROFILE_RELEASE_LTO=true \
    cargo build --target $1 --example intrinsics --release

# Ensure no references to any symbols from core
for rlib in $(echo $path); do
    set +ex
    echo "================================================================"
    echo checking $rlib for references to core
    echo "================================================================"

    $NM --quiet -U $rlib | grep 'T _ZN4core' | awk '{print $3}' | sort | uniq > defined_symbols.txt
    $NM --quiet -u $rlib | grep 'U _ZN4core' | awk '{print $2}' | sort | uniq > undefined_symbols.txt
    grep -v -F -x -f defined_symbols.txt undefined_symbols.txt

    if test $? = 0; then
        exit 1
    fi
    set -ex
done

true
