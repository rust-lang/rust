set -ex

. $(dirname $0)/env.sh

install_qemu() {
    case ${QEMU_ARCH:-$TRAVIS_OS_NAME} in
        i386)
            dpkg --add-architecture $QEMU_ARCH
            apt-get update
            apt-get install -y --no-install-recommends \
                    binfmt-support qemu-user-static:$QEMU_ARCH
            ;;
        linux)
            apt-get update
            apt-get install -y --no-install-recommends \
                    binfmt-support qemu-user-static
            ;;
    esac
}

install_binutils() {
    if [[ $TRAVIS_OS_NAME == "osx" ]]; then
        brew install binutils
    fi
}

install_rust() {
    if [[ $TRAVIS_OS_NAME == "osx" ]]; then
        curl https://sh.rustup.rs -sSf | sh -s -- -y --default-toolchain=nightly
    else
        rustup default nightly
    fi

    rustc -V
    cargo -V
}

add_rustup_target() {
    if [[ $TARGET != $HOST && ${CARGO:-cargo} == "cargo" ]]; then
        rustup target add $TARGET
    fi
}

install_xargo() {
    if [[ $CARGO == "xargo" ]]; then
        curl -sf "https://raw.githubusercontent.com/japaric/rust-everywhere/master/install.sh" | \
            bash -s -- --from japaric/xargo --at /root/.cargo/bin
    fi
}

main() {
    if [[ $TRAVIS_OS_NAME == "osx" || ${IN_DOCKER_CONTAINER:-n} == "y" ]]; then
        install_qemu
        install_binutils
        install_rust
        add_rustup_target
        install_xargo
    fi
}

main
