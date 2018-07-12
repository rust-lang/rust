set -euxo pipefail

main() {
    cargo run --package test-generator --target x86_64-unknown-linux-musl
    if cargo fmt --version >/dev/null 2>&1; then
        # nicer syntax error messages (if any)
        cargo fmt
    fi
    cross test --target $TARGET --release
}

main
