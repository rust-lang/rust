set -ex

. $(dirname $0)/env.sh

build() {
    ${CARGO:-cargo} build --target $TARGET
    ${CARGO:-cargo} build --target $TARGET --release
}

inspect() {
    $PREFIX$NM -g --defined-only target/**/debug/*.rlib
    set +e
    $PREFIX$OBJDUMP -Cd target/**/debug/*.rlib
    $PREFIX$OBJDUMP -Cd target/**/release/*.rlib
    set -e
}

run_tests() {
    if [[ $QEMU_LD_PREFIX ]]; then
        export RUST_TEST_THREADS=1
    fi

    if [[ $QEMU ]]; then
        cargo test --target $TARGET --no-run
        if [[ ${RUN_TESTS:-y} == "y" ]]; then
           $QEMU target/**/debug/rustc_builtins-*
        fi
        cargo test --target $TARGET --release --no-run
        if [[ ${RUN_TESTS:-y} == "y" ]]; then
            $QEMU target/**/release/rustc_builtins-*
        fi
    elif [[ ${RUN_TESTS:-y} == "y" ]]; then
        cargo test --target $TARGET
        cargo test --target $TARGET --release
    fi
}

main() {
    if [[ $DOCKER == "y" ]]; then
        local tag=2016-08-13

        docker run \
               --privileged \
               -e IN_DOCKER_CONTAINER=y \
               -e TARGET=$TARGET \
               -e TRAVIS_OS_NAME=$TRAVIS_OS_NAME \
               -v $(pwd):/mnt \
               japaric/rustc-builtins:$tag \
               sh -c 'set -ex;
                      cd /mnt;
                      export PATH="$PATH:/root/.cargo/bin";
                      bash ci/install.sh;
                      bash ci/script.sh'
    else
        build
        inspect
        run_tests
    fi
}

main
