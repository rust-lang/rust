set -ex

. $(dirname $0)/env.sh

install_binutils() {
    case $TRAVIS_OS_NAME in
        osx)
            brew install binutils
            ;;
        *)
            ;;
    esac
}

install_c_toolchain() {
    case $TARGET in
        aarch64-unknown-linux-gnu)
            sudo apt-get install -y --no-install-recommends \
                 gcc-aarch64-linux-gnu libc6-arm64-cross libc6-dev-arm64-cross
            ;;
        *)
            ;;
    esac
}

install_rust() {
    curl https://sh.rustup.rs -sSf | sh -s -- -y --default-toolchain=nightly

    rustc -V
    cargo -V
}

add_rustup_target() {
    if [[ $TARGET != $HOST ]]; then
        rustup target add $TARGET
    fi
}

configure_cargo() {
    if [[ $PREFIX ]]; then
        ${PREFIX}gcc -v

        mkdir -p .cargo
        cat >>.cargo/config <<EOF
[target.$TARGET]
linker = "${PREFIX}gcc"
EOF
    fi
}

main() {
    install_binutils
    install_c_toolchain
    install_rust
    add_rustup_target
    configure_cargo
}

main
