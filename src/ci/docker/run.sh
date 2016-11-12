#!/bin/sh
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

docker build \
  --rm \
  -t rust-ci \
  "`dirname "$script"`/$image"

mkdir -p $HOME/.ccache
mkdir -p $HOME/.cargo

exec docker run \
  --volume "$root_dir:/checkout:ro" \
  --workdir /tmp/obj \
  --env SRC=/checkout \
  --env CCACHE_DIR=/ccache \
  --volume "$HOME/.ccache:/ccache" \
  --env CARGO_HOME=/cargo \
  --env LOCAL_USER_ID=`id -u` \
  --volume "$HOME/.cargo:/cargo" \
  --interactive \
  --tty \
  rust-ci \
  /checkout/src/ci/run.sh
