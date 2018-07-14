set -euxo pipefail

main() {
    if ! hash cross >/dev/null 2>&1; then
        cargo install cross
    fi

    rustup component add rustfmt-preview

    rustup target add x86_64-unknown-linux-musl

    if [ $TARGET != x86_64-unknown-linux-gnu ]; then
        rustup target add $TARGET
    fi
}

main
