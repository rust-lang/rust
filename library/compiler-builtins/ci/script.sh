set -ex

. $(dirname $0)/env.sh

build() {
    cargo build --target $TARGET
    cargo build --target $TARGET --release
}

run_tests() {
    if [[ $QEMU_LD_PREFIX ]]; then
        export RUST_TEST_THREADS=1
    fi

    cargo test --target $TARGET
    cargo test --target $TARGET --release
}

inspect() {
    ${PREFIX}nm -g target/**/debug/*.rlib
    set +e
    ${PREFIX}objdump -Cd target/**/debug/*.rlib
    ${PREFIX}objdump -Cd target/**/release/*.rlib
    set -e
}

main() {
    build
    run_tests
    inspect
}

main
