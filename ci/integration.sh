#!/usr/bin/env bash

set -ex

: ${INTEGRATION?"The INTEGRATION environment variable must be set."}

# FIXME: this is causing the build to fail when rustfmt is found in .cargo/bin
# but cargo-fmt is not found.
#
# `which rustfmt` fails if rustfmt is not found. Since we don't install
# `rustfmt` via `rustup`, this is the case unless we manually install it. Once
# that happens, `cargo install --force` will be called, which installs
# `rustfmt`, `cargo-fmt`, etc to `~/.cargo/bin`. This directory is cached by
# travis (see `.travis.yml`'s "cache" key), such that build-bots that arrive
# here after the first installation will find `rustfmt` and won't need to build
# it again.
#
# which rustfmt || cargo install --force
cargo install --force

echo "Integration tests for: ${INTEGRATION}"

function check_fmt {
    cargo fmt --all -v -- --error-on-unformatted &> rustfmt_output
    if [[ $? != 0 ]]; then
        cat rustfmt_output
        return 1
    fi
    cat rustfmt_output
    ! cat rustfmt_output | grep -q "internal error"
    if [[ $? != 0 ]]; then
        return 1
    fi
    ! cat rustfmt_output | grep -q "warning"
    if [[ $? != 0 ]]; then
        return 1
    fi
    ! cat rustfmt_output | grep -q "Warning"
    if [[ $? != 0 ]]; then
        return 1
    fi
    cargo test --all
    if [[ $? != 0 ]]; then
        return $?
    fi
}

function check {
    cargo test --all
    if [[ $? != 0 ]]; then
        return 1
    fi
    check_fmt
    if [[ $? != 0 ]]; then
        return 1
    fi
}

case ${INTEGRATION} in
    cargo)
        git clone https://github.com/rust-lang/${INTEGRATION}.git
        cd ${INTEGRATION}
        export CFG_DISABLE_CROSS_TESTS=1
        check
        cd -
        ;;
    failure)
        git clone https://github.com/rust-lang-nursery/${INTEGRATION}.git
        cd ${INTEGRATION}/failure-1.X
        check
        cd -
        ;;
    *)
        git clone https://github.com/rust-lang-nursery/${INTEGRATION}.git
        cd ${INTEGRATION}
        check
        cd -
        ;;
esac
