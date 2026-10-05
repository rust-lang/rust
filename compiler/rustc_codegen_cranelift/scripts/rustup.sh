#!/usr/bin/env bash

set -e

export GIT_CONFIG_GLOBAL=$(pwd)/../rust/josh.gitconfig
export RUSTC_GIT=../rust

case $1 in
    "prepare")
        echo "=> Uninstalling all old nightlies"
        for nightly in $(RUSTUP_AUTO_INSTALL=0 rustup toolchain list | grep nightly | grep -v nightly-x86_64); do
            rustup toolchain uninstall "$nightly"
        done

        echo "=> Installing new nightly"
        rustup toolchain install

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
