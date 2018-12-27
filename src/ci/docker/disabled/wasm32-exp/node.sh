#!/usr/bin/env bash

path="$(dirname $1)"
file="$(basename $1)"

shift

cd "$path"
exec /node-v8.0.0-linux-x64/bin/node "$file" "$@"
