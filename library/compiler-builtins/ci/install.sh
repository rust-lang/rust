set -ex

. $(dirname $0)/env.sh

install_qemu() {
    if [[ $QEMU_LD_PREFIX ]]; then
        apt-get update
        apt-get install -y --no-install-recommends \
                binfmt-support qemu-user-static
    fi
}

install_gist() {
    if [[ $OSX ]]; then
        gem install gist -v 4.5.0
    fi
}

install_binutils() {
    if [[ $OSX ]]; then
        brew install binutils
    fi
}

install_rust() {
    if [[ $OSX ]]; then
        curl https://sh.rustup.rs -sSf | sh -s -- -y --default-toolchain=nightly
    else
        rustup default nightly
    fi

    rustup -V
    rustc -V
    cargo -V
}

add_rustup_target() {
    if [[ $TARGET != $HOST && $CARGO == cargo ]]; then
        rustup target add $TARGET
    fi
}

main() {
    if [[ $OSX || ${IN_DOCKER_CONTAINER:-n} == y ]]; then
        install_qemu
        install_gist
        install_binutils
        install_rust
        add_rustup_target
    fi
}

main
