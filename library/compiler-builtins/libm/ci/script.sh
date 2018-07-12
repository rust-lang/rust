set -euxo pipefail

main() {
    cargo run --package test-generator --target x86_64-unknown-linux-musl
    if hash cargo-fmt; then
        # nicer syntax error messages (if any)
        cargo fmt
    fi
    cross test --target $TARGET --release
}

main
