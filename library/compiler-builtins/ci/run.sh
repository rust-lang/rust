set -ex

# Test our implementation
case $1 in
    thumb*)
        for t in $(ls tests); do
            t=${t%.rs}

            # TODO(#154) enable these tests when aeabi_*mul are implemented
            case $t in
                powi*f2)
                    continue
                    ;;
            esac

            # FIXME(#150) debug assertion in divmoddi4
            case $1 in
                thumbv6m-*)
                    case $t in
                        divdi3 | divmoddi4 | moddi3 | modsi3 | udivmoddi4 | udivmodsi4 | umoddi3 | \
                            umodsi3)
                            continue
                            ;;
                    esac
                ;;
            esac

            xargo test --test $t --target $1 --features 'mem gen-tests' --no-run
            qemu-arm-static target/${1}/debug/$t-*

            xargo test --test $t --target $1 --features 'mem gen-tests' --no-run --release
            qemu-arm-static target/${1}/release/$t-*
        done
        ;;
    *)
        cargo test --no-default-features --features gen-tests --target $1
        cargo test --no-default-features --features 'gen-tests c' --target $1
        cargo test --no-default-features --features gen-tests --target $1 --release
        cargo test --no-default-features --features 'gen-tests c' --target $1 --release
        ;;
esac

# Verify that we haven't drop any intrinsic/symbol
case $1 in
    thumb*)
        xargo build --features c --target $1 --example intrinsics
        ;;
    *)
        cargo build --features c --target $1 --example intrinsics
        ;;
esac

# Verify that there are no undefined symbols to `panic` within our implementations
# TODO(#79) fix the undefined references problem for debug-assertions+lto
case $1 in
    thumb*)
        RUSTFLAGS="-C debug-assertions=no" xargo rustc --no-default-features --features c --target $1 --example intrinsics -- -C lto -C link-arg=-nostartfiles
        xargo rustc --no-default-features --features c --target $1 --example intrinsics --release -- -C lto
        ;;
    *)
        RUSTFLAGS="-C debug-assertions=no" cargo rustc --no-default-features --features c --target $1 --example intrinsics -- -C lto
        cargo rustc --no-default-features --features c --target $1 --example intrinsics --release -- -C lto
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

if [ $TRAVIS_OS_NAME = osx ]; then
    path=target/${1}/debug/deps/libcompiler_builtins-*.rlib
else
    path=/target/${1}/debug/deps/libcompiler_builtins-*.rlib
fi

for rlib in $(echo $path); do
    stdout=$($PREFIX$NM -g --defined-only $rlib)

    # NOTE On i586, It's normal that the get_pc_thunk symbol appears several times so ignore it
    set +e
    echo "$stdout" | sort | uniq -d | grep -v __x86.get_pc_thunk | grep 'T __'

    if test $? = 0; then
        exit 1
    fi
done

true
