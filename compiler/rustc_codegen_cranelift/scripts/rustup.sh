#!/usr/bin/env bash

set -e

TOOLCHAIN=${TOOLCHAIN:-$(date +%Y-%m-%d)}

function check_git_fixed_subtree() {
    if [[ ! -e ./git-fixed-subtree.sh ]]; then
        echo "Missing git-fixed-subtree.sh. Please run the following commands to download it:"
        echo "curl --proto '=https' --tlsv1.2 -sSf https://raw.githubusercontent.com/bjorn3/git/tqc-subtree-portable/contrib/subtree/git-subtree.sh -o git-fixed-subtree.sh"
        echo "chmod u+x git-fixed-subtree.sh"
        exit 1
    fi
    if [[ ! -x ./git-fixed-subtree.sh ]]; then
        echo "git-fixed-subtree.sh is not executable. Please run the following command to make it executable:"
        echo "chmod u+x git-fixed-subtree.sh"
        exit 1
    fi
}

case $1 in
    "prepare")
        echo "=> Installing new nightly"
        rustup toolchain install --profile minimal "nightly-${TOOLCHAIN}" # Sanity check to see if the nightly exists
        sed -i "s/\"nightly-.*\"/\"nightly-${TOOLCHAIN}\"/" rust-toolchain
        rustup component add rustfmt || true

        echo "=> Uninstalling all old nightlies"
        for nightly in $(rustup toolchain list | grep nightly | grep -v "$TOOLCHAIN" | grep -v nightly-x86_64); do
            rustup toolchain uninstall "$nightly"
        done

        ./clean_all.sh

        ./y.sh prepare
        ;;
    "commit")
        git add rust-toolchain
        git commit -m "Rustup to $(rustc -V)"
        ;;
    "push")
        check_git_fixed_subtree

        cg_clif=$(pwd)
        pushd ../rust
        git pull origin master
        branch=sync_cg_clif-$(date +%Y-%m-%d)
        git checkout -b "$branch"
        "$cg_clif/git-fixed-subtree.sh" pull --prefix=compiler/rustc_codegen_cranelift/ https://github.com/rust-lang/rustc_codegen_cranelift.git master
        git push -u my "$branch"

        # immediately merge the merge commit into cg_clif to prevent merge conflicts when syncing
        # from rust-lang/rust later
        "$cg_clif/git-fixed-subtree.sh" push --prefix=compiler/rustc_codegen_cranelift/ "$cg_clif" sync_from_rust
        popd
        git merge sync_from_rust
	;;
    "pull")
        check_git_fixed_subtree

        RUST_VERS=$(curl "https://static.rust-lang.org/dist/$TOOLCHAIN/channel-rust-nightly-git-commit-hash.txt")
        echo "Pulling $RUST_VERS ($TOOLCHAIN)"

        cg_clif=$(pwd)
        pushd ../rust
        git fetch origin master
        git -c advice.detachedHead=false checkout "$RUST_VERS"
        "$cg_clif/git-fixed-subtree.sh" push --prefix=compiler/rustc_codegen_cranelift/ "$cg_clif" sync_from_rust
        popd
        git merge sync_from_rust -m "Sync from rust $RUST_VERS"
        git branch -d sync_from_rust
        ;;
    *)
        echo "Unknown command '$1'"
        echo "Usage: ./rustup.sh prepare|commit"
        ;;
esac
