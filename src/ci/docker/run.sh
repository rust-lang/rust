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

set -e

script=`cd $(dirname $0) && pwd`/`basename $0`
image=$1

docker_dir="`dirname $script`"
ci_dir="`dirname $docker_dir`"
src_dir="`dirname $ci_dir`"
root_dir="`dirname $src_dir`"

source "$ci_dir/shared.sh"

if [ -f "$docker_dir/$image/Dockerfile" ]; then
    retry docker \
      build \
      --rm \
      -t rust-ci \
      -f "$docker_dir/$image/Dockerfile" \
      "$docker_dir"
elif [ -f "$docker_dir/disabled/$image/Dockerfile" ]; then
    if [ -n "$TRAVIS_OS_NAME" ]; then
        echo Cannot run disabled images on travis!
        exit 1
    fi
    retry docker \
      build \
      --rm \
      -t rust-ci \
      -f "$docker_dir/disabled/$image/Dockerfile" \
      "$docker_dir"
else
    echo Invalid image: $image
    exit 1
fi

objdir=$root_dir/obj

mkdir -p $HOME/.cargo
mkdir -p $objdir/tmp

args=
if [ "$SCCACHE_BUCKET" != "" ]; then
    args="$args --env SCCACHE_BUCKET=$SCCACHE_BUCKET"
    args="$args --env AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID"
    args="$args --env AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY"
    args="$args --env SCCACHE_ERROR_LOG=/tmp/sccache/sccache.log"
    args="$args --volume $objdir/tmp:/tmp/sccache"
else
    mkdir -p $HOME/.cache/sccache
    args="$args --env SCCACHE_DIR=/sccache --volume $HOME/.cache/sccache:/sccache"
fi

exec docker \
  run \
  --volume "$root_dir:/checkout:ro" \
  --volume "$objdir:/checkout/obj" \
  --workdir /checkout/obj \
  --env SRC=/checkout \
  $args \
  --env CARGO_HOME=/cargo \
  --env DEPLOY=$DEPLOY \
  --env DEPLOY_ALT=$DEPLOY_ALT \
  --env LOCAL_USER_ID=`id -u` \
  --volume "$HOME/.cargo:/cargo" \
  --volume "$HOME/rustsrc:$HOME/rustsrc" \
  --privileged \
  --rm \
  rust-ci \
  /checkout/src/ci/run.sh
