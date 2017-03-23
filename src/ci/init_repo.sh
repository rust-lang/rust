#!/bin/bash
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

set -o xtrace

ci_dir=$(cd $(dirname $0) && pwd)
. "$ci_dir/shared.sh"

REPO_DIR="$1"
CACHE_DIR="$2"

cache_src_dir="$CACHE_DIR/src"
# If the layout of the cache directory changes, bump the number here
# (and anywhere else this file is referenced) so the cache is wiped
cache_valid_file="$CACHE_DIR/cache_valid1"

if [ ! -d "$REPO_DIR" -o ! -d "$REPO_DIR/.git" ]; then
    echo "Error: $REPO_DIR does not exist or is not a git repo"
    exit 1
fi
cd $REPO_DIR
if [ ! -d "$CACHE_DIR" ]; then
    echo "Error: $CACHE_DIR does not exist or is not an absolute path"
    exit 1
fi

# Wipe the cache if it's not valid, or mark it as invalid while we update it
if [ ! -f "$cache_valid_file" ]; then
    rm -rf "$CACHE_DIR" && mkdir "$CACHE_DIR"
else
    rm "$cache_valid_file"
fi

# Update the cache (a pristine copy of the rust source master)
if [ ! -d "$cache_src_dir/.git" ]; then
    retry sh -c "rm -rf $cache_src_dir && mkdir -p $cache_src_dir && \
        git clone https://github.com/rust-lang/rust.git $cache_src_dir"
fi
retry sh -c "cd $cache_src_dir && git reset --hard && git pull"
retry sh -c "cd $cache_src_dir && \
    git submodule deinit -f . && git submodule sync && git submodule update --init"

# Cache was updated without errors, mark it as valid
touch "$cache_valid_file"

# Update the submodules of the repo we're in, using the pristine repo as
# a cache for any object files
# No, `git submodule foreach` won't work:
# http://stackoverflow.com/questions/12641469/list-submodules-in-a-git-repository
modules="$(git config --file .gitmodules --get-regexp '\.path$' | cut -d' ' -f2)"
for module in $modules; do
    if [ ! -d "$cache_src_dir/$module" ]; then
        echo "WARNING: $module not found in pristine repo"
        retry sh -c "git submodule deinit -f $module && git submodule update --init $module"
        continue
    fi
    retry sh -c "git submodule deinit -f $module && \
        git submodule update --init --reference $cache_src_dir/$module $module"
done
