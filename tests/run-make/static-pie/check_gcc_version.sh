#!/bin/bash

set -euo pipefail

if command -v gcc > /dev/null
then
  GCC_VERSION=$(echo __GNUC__ | gcc -E -x c - | grep -v -e '^#' )
  echo "gcc version $GCC_VERSION detected"
  if (( $GCC_VERSION >= 8 ))
  then
    echo "gcc supports -static-pie"
    exit 0
  else
    echo "gcc too old to support -static-pie, skipping test"
    exit 1
  fi
else
  echo "No gcc version detected"
  exit 2
fi
