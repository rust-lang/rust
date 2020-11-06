#!/usr/bin/env bash

set -euo pipefail

indir="${1:?Missing argument 1: input directory}"

tidy () {
  command tidy \
      --indent yes \
      --indent-spaces 2 \
      --wrap 0 \
      --show-warnings no \
      --markup yes \
      --quiet yes \
      "$@" \
      >/dev/null \
  # tidy exits with code 1 if there were any warnings
  || [ $? -eq 1 ]
}

find "$indir" -type f -name '*.html' -print0 \
| while IFS= read -d '' -r file
do
  tidy -modify "$file"
done
