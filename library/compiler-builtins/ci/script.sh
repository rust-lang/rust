set -ex

. $(dirname $0)/env.sh

gist() {
    wgetpaste -s gists -d "'$1' from commit '$TRAVIS_COMMIT' on branch '$TRAVIS_BRANCH'"
    echo "Disassembly available at the above URL."
}

build() {
    ${CARGO:-cargo} build --target $TARGET
    ${CARGO:-cargo} build --target $TARGET --release
}

inspect() {
    $PREFIX$NM -g --defined-only target/**/debug/*.rlib

    set +e
    $PREFIX$OBJDUMP -Cd target/**/release/*.rlib | gist "$TARGET/rustc-builtins.rlib"
    set -e

    # Check presence of weak symbols
    case $TRAVIS_OS_NAME in
        linux)
            local symbols=( memcmp memcpy memmove memset )
            for symbol in "${symbols[@]}"; do
                $PREFIX$NM target/**/debug/deps/librlibc*.rlib | grep -q "W $symbol"
            done
            ;;
    esac
}

run_tests() {
    if [[ $QEMU_LD_PREFIX ]]; then
        export RUST_TEST_THREADS=1
    fi

    if [[ ${RUN_TESTS:-y} == "y" ]]; then
        cargo test --target $TARGET
        cargo test --target $TARGET --release
    fi
}

main() {
    if [[ $TRAVIS_OS_NAME == "linux" && ${IN_DOCKER_CONTAINER:-n} == "n" ]]; then
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
