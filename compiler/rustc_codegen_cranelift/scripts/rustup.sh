#!/usr/bin/env bash

set -e

export GIT_CONFIG_GLOBAL=$(pwd)/../rust/josh.gitconfig
export RUSTC_GIT=../rust

case $1 in
    "push")
        username=${2:-bjorn3}
        branch=sync_cg_clif-$(date +%Y-%m-%d)
        rustc-josh-sync push "$branch" "$username"
	;;
    "pull")
        echo "=> Uninstalling all old nightlies"
        for nightly in $(RUSTUP_AUTO_INSTALL=0 rustup toolchain list | grep nightly | grep -v nightly-x86_64); do
            rustup toolchain uninstall "$nightly"
        done

        echo "=> Cleaning build artifacts"
        ./clean_all.sh

        rustc-josh-sync pull

        echo "=> Installing new nightly"
        rustup toolchain install

        echo "=> Fetching deps"
        ./y.sh prepare
        ;;
    *)
        echo "Unknown command '$1'"
        echo "Usage: ./rustup.sh pull|push [fork]"
        ;;
esac
