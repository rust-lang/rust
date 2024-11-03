#!/bin/bash
ORIGIN="https://github.com/rust-lang"
LLVM_FORK="https://github.com/borrow-sanitizer/llvm-project.git"
RUST_UPSTREAM="$ORIGIN/rust.git"
LLVM_UPSTREAM="$ORIGIN/llvm-project.git"
MAIN_BRANCH="bsan"
BASE_TAG="base"
BASE_COMMIT=$(git rev-list -n 1 $BASE_TAG)
git checkout "$MAIN_BRANCH" || exit

init_upstream() {
    local upstream=$1
    (git remote -v | grep -w upstream && \
    git remote set-url upstream "$upstream") || \
    git remote add upstream "$upstream"
    git fetch --no-tags --recurse-submodules=no upstream 
}

rustup update nightly 
rustup default nightly

NIGHTLY_COMMIT_HASH=$(rustc --version | cut -d ' ' -f 3 | cut -d '(' -f 2)
DATE=$(rustc --version | cut -d ' ' -f 4 | cut -d ')' -f 1)

if [[ ! $NIGHTLY_COMMIT_HASH =~ ^[a-z0-9]{9}$ ]]; then
    echo "Error: invalid commit hash format: '$NIGHTLY_COMMIT_HASH'"
    exit
fi

if [[ ! $DATE =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
    echo "Error: invalid date format: '$DATE'"
    exit
fi

NIGHTLY_TOOLCHAIN="nightly-$DATE"
git submodule update --init --no-recommend-shallow src/llvm-project 
(cd src/llvm-project && init_upstream $LLVM_UPSTREAM)
init_upstream $RUST_UPSTREAM
git branch -D nightly
git checkout -b nightly "$NIGHTLY_COMMIT_HASH" || exit
git submodule update --rebase src/llvm-project
git submodule set-url -- src/llvm-project $LLVM_FORK
git submodule set-branch --branch $MAIN_BRANCH -- src/llvm-project 
git add src/llvm-project .gitmodules
git commit -m "$NIGHTLY_TOOLCHAIN"
git tag -D $BASE_TAG
git tag -a $BASE_TAG -m "$NIGHTLY_TOOLCHAIN"
git checkout "$MAIN_BRANCH" || exit
git rebase --onto nightly "$BASE_COMMIT" $MAIN_BRANCH || exit