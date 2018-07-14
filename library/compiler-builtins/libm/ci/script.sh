set -euxo pipefail

main() {
    if [ $TARGET = cargo-fmt ]; then
        cargo fmt -- --check
        return
    fi

    # quick check
    cargo check

    # check that we can source import libm into compiler-builtins
    cargo check --package cb

    # generate tests
    cargo run --package test-generator --target x86_64-unknown-linux-musl

    # nicer syntax error messages (if any)
    cargo fmt

    # run tests
    cross test --target $TARGET --release

    # TODO need to fix overflow issues (cf. issue #4)
    # cross test --target $TARGET
}

main
