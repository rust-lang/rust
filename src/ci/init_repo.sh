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

travis_fold start update_cache
travis_time_start

# Update the cache (a pristine copy of the rust source master)
retry sh -c "rm -rf $cache_src_dir && mkdir -p $cache_src_dir && \
    git clone --depth 1 https://github.com/rust-lang/rust.git $cache_src_dir"
(cd $cache_src_dir && git rm src/llvm)
retry sh -c "cd $cache_src_dir && \
    git submodule deinit -f . && git submodule sync && git submodule update --init"

travis_fold end update_cache
travis_time_finish

travis_fold start update_submodules
travis_time_start

# Update the submodules of the repo we're in, using the pristine repo as
# a cache for any object files
# No, `git submodule foreach` won't work:
# http://stackoverflow.com/questions/12641469/list-submodules-in-a-git-repository
modules="$(git config --file .gitmodules --get-regexp '\.path$' | cut -d' ' -f2)"
for module in $modules; do
    if [ "$module" = src/llvm ]; then
        commit="$(git ls-tree HEAD src/llvm | awk '{print $3}')"
        git rm src/llvm
        retry sh -c "rm -f $commit.tar.gz && \
            curl -sSL -O https://github.com/rust-lang/llvm/archive/$commit.tar.gz"
        tar -C src/ -xf "$commit.tar.gz"
        rm "$commit.tar.gz"
        mv "src/llvm-$commit" src/llvm
        continue
    fi
    if [ ! -e "$cache_src_dir/$module/.git" ]; then
        echo "WARNING: $module not found in pristine repo"
        retry sh -c "git submodule deinit -f $module && \
            git submodule update --init --recursive $module"
        continue
    fi
    retry sh -c "git submodule deinit -f $module && \
        git submodule update --init --recursive --reference $cache_src_dir/$module $module"
done

travis_fold end update_submodules
travis_time_finish

travis_fold end init_repo
