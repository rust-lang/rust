set -ex

. $(dirname $0)/env.sh

install_qemu() {
    case $TRAVIS_OS_NAME in
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

install_wgetpaste() {
    if [[ $TRAVIS_OS_NAME == "osx" ]]; then
        brew install wgetpaste
    else
        curl -O http://wgetpaste.zlin.dk/wgetpaste-2.28.tar.bz2
        tar -xvf wgetpaste-2.28.tar.bz2
        sudo mv ./wgetpaste-2.28/wgetpaste /usr/bin
        rm -r wgetpaste-2.28*
    fi
}

main() {
    if [[ $TRAVIS_OS_NAME == "osx" || ${IN_DOCKER_CONTAINER:-n} == "y" ]]; then
        install_qemu
        install_binutils
        install_rust
        add_rustup_target
        install_xargo
        install_wgetpaste
    fi
}

main
