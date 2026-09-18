#!/usr/bin/env bash

set -e

TOOLCHAIN=${TOOLCHAIN:-$(date +%Y-%m-%d)}

export GIT_CONFIG_GLOBAL=../rust/josh.gitconfig
export RUSTC_GIT=../rust

case $1 in
    "prepare")
        echo "=> Installing new nightly"
        rustup toolchain install --profile minimal "nightly-${TOOLCHAIN}" # Sanity check to see if the nightly exists
        sed -i "s/\"nightly-.*\"/\"nightly-${TOOLCHAIN}\"/" rust-toolchain.toml

        echo "=> Uninstalling all old nightlies"
        for nightly in $(rustup toolchain list | grep nightly | grep -v "$TOOLCHAIN" | grep -v nightly-x86_64); do
            rustup toolchain uninstall "$nightly"
        done

        ./clean_all.sh

        ./y.sh prepare
        ;;
    "push")
        username=${2:-bjorn3}
        branch=sync_cg_clif-$(date +%Y-%m-%d)
        rustc-josh-sync push "$branch" "$username"
	;;
    "pull")
        git checkout -b sync_from_rust
        rustc-josh-sync pull
        ;;
    *)
        echo "Unknown command '$1'"
        echo "Usage: ./rustup.sh prepare|pull|push [fork]"
        ;;
esac
