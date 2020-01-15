#!/bin/bash

if [ -z $CHANNEL ]; then
export CHANNEL='debug'
fi

pushd $(dirname "$0") >/dev/null
source config.sh

# read nightly compiler from rust-toolchain file
TOOLCHAIN=$(cat rust-toolchain)

popd >/dev/null

cmd=$1
shift

cargo +${TOOLCHAIN} $cmd --target $TARGET_TRIPLE $@
