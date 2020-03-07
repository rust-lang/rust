#!/usr/bin/env sh

set -ex

retry() {
    result=0
    count=1
    max=5
    while [ "$count" -le 3 ]; do
        [ "$result" -ne 0 ] && {
            printf "\nRetrying, %d of %d\n" $count $max >&2
        }
        "$@"
        result=$?
        [ $result -eq 0 ] && break
        count=$((count + 1))
        sleep 1
    done

    [ "$count" -gt 3 ] && {
        printf "\nFailed %d times.\n" $max >&2
    }

    return $result
}


if retry rustup component add rustfmt ; then
    cargo fmt --all -- --check
fi

if retry rustup component add clippy ; then
    cargo clippy --all -- -D clippy::pedantic
fi

if [ "${TRAVIS_OS_NAME}" = "linux" ]; then
    if retry rustup component add clippy ; then
        cargo clippy --all --target=i586-unknown-linux-gnu -- -D clippy::all -D clippy::pedantic
    fi
fi

if command -v shellcheck ; then
    shellcheck --version
    shellcheck ci/*.sh
fi
