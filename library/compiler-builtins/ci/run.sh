# Test our implementation
case $1 in
    thumb*)
        curl -sf "https://raw.githubusercontent.com/japaric/rust-everywhere/master/install.sh" | \
            bash -s -- --at /usr/bin --from japaric/xargo --tag v0.1.10
        xargo build --target $1
        xargo build --target $1 --release
        ;;
    # QEMU crashes even when executing the simplest cross compiled C program:
    # `int main() { return 0; }`
    powerpc64le-unknown-linux-gnu)
        cargo test --target $1 --no-run
        cargo test --target $1 --no-run --release
        ;;
    *)
        cargo test --target $1
        cargo test --target $1 --release
        ;;
esac

# Verify that we haven't drop any intrinsic/symbol
case $1 in
    thumb*)
        xargo build --features c --target $1 --bin intrinsics
        ;;
    *)
        cargo build --features c --target $1 --bin intrinsics
        ;;
esac

# Look out for duplicated symbols when we include the compiler-rt (C) implementation
PREFIX=$(echo $1 | sed -e 's/unknown-//')
case $1 in
    armv7-*)
        PREFIX=arm-linux-gnueabihf-
        ;;
    thumb*)
        PREFIX=arm-none-eabi-
        ;;
    *-unknown-linux-gnu | *-apple-darwin)
        PREFIX=
        ;;
esac

case $TRAVIS_OS_NAME in
    osx)
        NM=gnm

        # NOTE OSx's nm doesn't accept the `--defined-only` or provide an equivalent.
        # Use GNU nm instead
        brew install binutils
        ;;
    *)
        NM=nm
        ;;
esac

$PREFIX$NM -g --defined-only /tmp/target/${1}/debug/librustc_builtins.rlib | \
    sort | uniq -d | grep 'T __'

if test $? = 0; then
    exit 1
fi
