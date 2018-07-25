set -euxo pipefail

main() {
    if [ $TARGET = cargo-fmt ]; then
        cargo fmt -- --check
        return
    fi

    # test that the functions don't contain invocations of `panic!`
    if [ $TRAVIS_RUST_VERSION ]; then
        cross build --release --target $TARGET --example no-panic
        return
    fi

    # quick check
    cargo check

    # check that we can source import libm into compiler-builtins
    cargo check --package cb

    # run unit tests
    cross test --lib --features checked --target $TARGET --release

    # generate tests
    cargo run --package test-generator --target x86_64-unknown-linux-musl

    # run generated tests
    cross test --tests --features checked --target $TARGET --release

    # TODO need to fix overflow issues (cf. issue #4)
    # cross test --target $TARGET
}

main
