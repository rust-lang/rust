#!/bin/bash

set -e

case $1 in
    "prepare")
        # FIXME Automatically detect latest nightly
        read -p "Date of nightly to use (aka yesterday GMT): " TOOLCHAIN

        echo "=> Installing new nightly"
        rustup toolchain install --profile minimal nightly-${TOOLCHAIN} # Sanity check to see if the nightly exists
        echo nightly-${TOOLCHAIN} > rust-toolchain

        echo "=> Uninstalling all old nighlies"
        for nightly in $(rustup toolchain list | grep nightly- | grep -v $TOOLCHAIN); do
            rustup toolchain uninstall $nightly
        done

        ./clean_all.sh
        ./prepare.sh
        ;;
    "commit")
        git commit -m "$(rustc -V)"
        ;;
    *)
        echo "Unknown command '$1'"
        echo "Usage: ./rustup.sh prepare|commit"
        ;;
esac
