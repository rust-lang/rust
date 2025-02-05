#!/bin/bash

set -eou pipefail

git fetch origin
pushd $2
git fetch origin
popd

if [ "$(git rev-parse --show-prefix)" != "" ]; then
    echo "Run this script from the git root" >&2
    exit 1
fi

if [ "$(git rev-parse HEAD)" != "$(git rev-parse origin/master)" ]; then
    echo "$(pwd) is not at origin/master" >&2
    exit 1
fi

if [ ! -f library/portable-simd/git-subtree.sh ]; then
    curl -sS https://raw.githubusercontent.com/bjorn3/git/tqc-subtree-portable/contrib/subtree/git-subtree.sh -o library/portable-simd/git-subtree.sh
    chmod +x library/portable-simd/git-subtree.sh
fi

today=$(date +%Y-%m-%d)

case $1 in
    "push")
        upstream=rust-upstream-$today
        merge=sync-from-rust-$today

        pushd $2
        git checkout master
        git pull
        popd

        library/portable-simd/git-subtree.sh push -P library/portable-simd $2 $upstream

        pushd $2
        git checkout -B $merge origin/master
        git merge $upstream
        popd
        echo "Branch \`$merge\` created in \`$2\`. You may need to resolve merge conflicts."
        ;;
    "pull")
        branch=sync-from-portable-simd-$today

        git checkout -B $branch
        echo "Creating branch \`$branch\`... You may need to resolve merge conflicts."
        library/portable-simd/git-subtree.sh pull -P library/portable-simd $2 origin/master
        ;;
esac
