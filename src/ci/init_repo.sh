#!/usr/bin/env bash
# Copyright 2016 The Rust Project Developers. See the COPYRIGHT
# file at the top-level directory of this distribution and at
# http://rust-lang.org/COPYRIGHT.
#
# Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
# http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
# <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
# option. This file may not be copied, modified, or distributed
# except according to those terms.

set -o errexit
set -o pipefail
set -o nounset

ci_dir=$(cd $(dirname $0) && pwd)
. "$ci_dir/shared.sh"

travis_fold start init_repo
travis_time_start

REPO_DIR="$1"
CACHE_DIR="$2"

cache_src_dir="$CACHE_DIR/src"

if [ ! -d "$REPO_DIR" -o ! -d "$REPO_DIR/.git" ]; then
    echo "Error: $REPO_DIR does not exist or is not a git repo"
    exit 1
fi
cd $REPO_DIR
if [ ! -d "$CACHE_DIR" ]; then
    echo "Error: $CACHE_DIR does not exist or is not an absolute path"
    exit 1
fi

rm -rf "$CACHE_DIR"
mkdir "$CACHE_DIR"

# On the beta channel we'll be automatically calculating the prerelease version
# via the git history, so unshallow our shallow clone from CI.
if grep -q RUST_RELEASE_CHANNEL=beta src/ci/run.sh; then
  git fetch origin --unshallow beta master
fi

function fetch_submodule {
    local module=$1
    local cached="download-${module//\//-}.tar.gz"
    retry sh -c "rm -f $cached && \
        curl -sSL -o $cached $2"
    mkdir $module
    touch "$module/.git"
    tar -C $module --strip-components=1 -xf $cached
    rm $cached
}

included="src/llvm src/llvm-emscripten src/doc/book src/doc/rust-by-example"
modules="$(git config --file .gitmodules --get-regexp '\.path$' | cut -d' ' -f2)"
modules=($modules)
use_git=""
urls="$(git config --file .gitmodules --get-regexp '\.url$' | cut -d' ' -f2)"
urls=($urls)
for i in ${!modules[@]}; do
    module=${modules[$i]}
    if [[ " $included " = *" $module "* ]]; then
        commit="$(git ls-tree HEAD $module | awk '{print $3}')"
        git rm $module
        url=${urls[$i]}
        url=${url/\.git/}
        fetch_submodule $module "$url/archive/$commit.tar.gz" &
        continue
    else
        use_git="$use_git $module"
    fi
done
retry sh -c "git submodule deinit -f $use_git && \
    git submodule sync && \
    git submodule update -j 16 --init --recursive $use_git"
wait
travis_fold end init_repo
travis_time_finish
