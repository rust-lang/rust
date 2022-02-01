#!/bin/bash

set -euo pipefail

if command -v clang > /dev/null
then
  CLANG_VERSION=$(echo __clang_major__ | clang -E -x c - | grep -v -e '^#' )
  echo "clang version $CLANG_VERSION detected"
  if (( $CLANG_VERSION >= 9 ))
  then
    echo "clang supports -static-pie"
    exit 0
  else
    echo "clang too old to support -static-pie, skipping test"
    exit 1
  fi
else
  echo "No clang version detected"
  exit 2
fi
