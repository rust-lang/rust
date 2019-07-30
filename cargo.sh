#!/bin/bash

if [ -z $CHANNEL ]; then
export CHANNEL='debug'
fi

pushd $(dirname "$0") >/dev/null
source config.sh
popd >/dev/null

cmd=$1
shift

cargo $cmd --target $TARGET_TRIPLE $@
