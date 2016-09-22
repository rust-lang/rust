set -ex

. $(dirname $0)/env.sh

gist_it() {
    gist -d "'$TARGET/rustc-builtins.rlib' from commit '$TRAVIS_COMMIT' on branch '$TRAVIS_BRANCH'"
    echo "Disassembly available at the above URL."
}

build() {
    if [[ $WEAK ]]; then
        $CARGO build --features weak --target $TARGET
        $CARGO build --features weak --target $TARGET --release
    else
        $CARGO build --target $TARGET
        $CARGO build --target $TARGET --release
    fi
}

inspect() {
    $PREFIX$NM -g --defined-only target/**/debug/*.rlib

    set +e
    $PREFIX$OBJDUMP -Cd target/**/release/*.rlib | gist_it
    set -e

    # Check presence/absence of weak symbols
    if [[ $WEAK ]]; then
        local symbols=( memcmp memcpy memmove memset )
        for symbol in "${symbols[@]}"; do
            $PREFIX$NM target/$TARGET/debug/deps/librlibc-*.rlib | grep -q "W $symbol"
        done
    else
        set +e
        ls target/$TARGET/debug/deps/librlibc-*.rlib

        if [[ $? == 0 ]]; then
            exit 1
        fi

        set -e
    fi

}

run_tests() {
    if [[ $QEMU_LD_PREFIX ]]; then
        export RUST_TEST_THREADS=1
    fi

    if [[ $RUN_TESTS == y ]]; then
        cargo test --target $TARGET
        cargo test --target $TARGET --release
    fi
}

main() {
    if [[ $LINUX && ${IN_DOCKER_CONTAINER:-n} == n ]]; then
        # NOTE The Dockerfile of this image is in the docker branch of this repository
        docker run \
               --privileged \
               -e IN_DOCKER_CONTAINER=y \
               -e TARGET=$TARGET \
               -e TRAVIS_BRANCH=$TRAVIS_BRANCH \
               -e TRAVIS_COMMIT=$TRAVIS_COMMIT \
               -e TRAVIS_OS_NAME=$TRAVIS_OS_NAME \
               -e WEAK=$WEAK \
               -v $(pwd):/mnt \
               japaric/rustc-builtins \
               sh -c 'cd /mnt;
                      bash ci/install.sh;
                      bash ci/script.sh'
    else
        build
        inspect
        run_tests
    fi
}

main
