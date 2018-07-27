set -euxo pipefail

main() {
    if [ $TARGET = cargo-fmt ]; then
        rustup component add rustfmt-preview
        return
    fi

    if ! hash cross >/dev/null 2>&1; then
        cargo install cross
    fi

    rustup target add x86_64-unknown-linux-musl

    if [ $TARGET != x86_64-unknown-linux-gnu ]; then
        rustup target add $TARGET
    fi

    mkdir -p ~/.local/bin
    curl -L https://github.com/japaric/qemu-bin/raw/master/14.04/qemu-arm-2.12.0 > ~/.local/bin/qemu-arm
    chmod +x ~/.local/bin/qemu-arm
    qemu-arm --version
}

main
