case $TRAVIS_OS_NAME in
    linux)
        HOST=x86_64-unknown-linux-gnu
        NM=nm
        OBJDUMP=objdump
        LINUX=y
        ;;
    osx)
        HOST=x86_64-apple-darwin
        NM=gnm
        OBJDUMP=gobjdump
        OSX=y
        ;;
esac

# NOTE For rustup
export PATH="$HOME/.cargo/bin:$PATH"

CARGO=cargo
RUN_TESTS=y

# NOTE For the host and its 32-bit variants we don't need prefixed tools or QEMU
if [[ $TARGET != $HOST && ! $TARGET =~ ^i.86- ]]; then
    GCC_TRIPLE=${TARGET//unknown-/}

    case $TARGET in
        armv7-unknown-linux-gnueabihf)
            GCC_TRIPLE=arm-linux-gnueabihf
            ;;
        powerpc64le-unknown-linux-gnu)
            # QEMU crashes even when executing the simplest cross compiled C program:
            # `int main() { return 0; }`
            RUN_TESTS=n
            ;;
        thumbv*-none-eabi)
            CARGO=xargo
            GCC_TRIPLE=arm-none-eabi
            # Bare metal targets. No `std` or `test` crates for these targets.
            RUN_TESTS=n
            ;;
    esac

    if [[ $RUN_TESTS == y ]]; then
        # NOTE(export) so this can reach the processes that `cargo test` spawns
        export QEMU_LD_PREFIX=/usr/$GCC_TRIPLE
    fi

    PREFIX=$GCC_TRIPLE-
fi
