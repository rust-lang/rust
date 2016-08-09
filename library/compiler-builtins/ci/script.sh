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

    if [[ $QEMU ]]; then
        cargo test --target $TARGET --no-run
        $QEMU target/**/debug/rustc_builtins-*
        cargo test --target $TARGET --release --no-run
        $QEMU target/**/release/rustc_builtins-*
    else
        cargo test --target $TARGET
        cargo test --target $TARGET --release
    fi
}

inspect() {
    $PREFIX$NM -g --defined-only target/**/debug/*.rlib
    set +e
    $PREFIX$OBJDUMP -Cd target/**/debug/*.rlib
    $PREFIX$OBJDUMP -Cd target/**/release/*.rlib
    set -e
}

main() {
    if [[ $DOCKER == "y" ]]; then
        docker run \
               -e DOCKER=n \
               -e TARGET=$TARGET \
               -e TRAVIS_OS_NAME=$TRAVIS_OS_NAME \
               -v $(pwd):/mnt \
               ubuntu:16.04 \
               sh -c 'set -ex;
                      cd /mnt;
                      export PATH="$PATH:$HOME/.cargo/bin";
                      bash ci/install.sh;
                      bash ci/script.sh'
    else
        build
        run_tests
        inspect
    fi
}

main
