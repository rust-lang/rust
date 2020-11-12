#!/bin/bash

set -e

case $1 in
    "prepare")
        TOOLCHAIN=$(date +%Y-%m-%d)

        echo "=> Installing new nightly"
        rustup toolchain install --profile minimal nightly-${TOOLCHAIN} # Sanity check to see if the nightly exists
        echo nightly-${TOOLCHAIN} > rust-toolchain
        rustup component add rustfmt || true

        echo "=> Uninstalling all old nighlies"
        for nightly in $(rustup toolchain list | grep nightly | grep -v $TOOLCHAIN | grep -v nightly-x86_64); do
            rustup toolchain uninstall $nightly
        done

        ./clean_all.sh
        ./prepare.sh

        (cd build_sysroot && cargo update)

        ;;
    "commit")
        git add rust-toolchain build_sysroot/Cargo.lock
        git commit -m "Rustup to $(rustc -V)"
        ;;
    "push")
	cg_clif=$(pwd)
	pushd ../rust
	branch=update_cg_clif-$(date +%Y-%m-%d)
	git checkout -b $branch
	git subtree pull --prefix=compiler/rustc_codegen_cranelift/ https://github.com/bjorn3/rustc_codegen_cranelift.git master
	git push -u my $branch
	popd
	;;
    *)
        echo "Unknown command '$1'"
        echo "Usage: ./rustup.sh prepare|commit"
        ;;
esac
