#!/usr/bin/env bash

set -euo pipefail

indir="${1:?Missing argument 1: input directory}"

tidy () {
  {
    # new-inline-tags is workaround for:
    #   https://github.com/rust-lang/stdarch/issues/945
    #   https://github.com/rust-lang/mdBook/issues/1372
    command tidy \
        --indent yes \
        --indent-spaces 2 \
        --wrap 0 \
        --show-warnings no \
        --markup yes \
        --quiet yes \
        --new-inline-tags 'c t' \
        "$@" \
        >/dev/null \
    || {
      # tidy exits with code 1 if there were any warnings :facepalm:
      status=$?
      if [ $status != 0 ] && [ $status != 1 ]
      then
        echo "While tidying $1" >&2
        exit 1
      fi
    }
  } | sed -E 's/#[0-9]+(-[0-9]+)?/#line/g'
}

find "$indir" -type f -name '*.html' -print0 \
| while IFS= read -d '' -r file
do
  tidy -modify "$file"
done
