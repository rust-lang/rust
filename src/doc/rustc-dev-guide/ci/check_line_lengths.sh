#!/bin/bash

if [ "$1" == "--help" ]; then
    echo 'Usage:'
    echo '  MAX_LINE_LENGTH=100' "$0" 'src/**/*.md'
    exit 1
fi

if [ "$MAX_LINE_LENGTH" == "" ]; then
    echo '`MAX_LINE_LENGTH` environment variable not set. Try --help.'
    exit 1
fi

if [ "$1" == "" ]; then
    echo 'No files provided.'
    exit 1
fi

echo "Checking line lengths in all source files <= $MAX_LINE_LENGTH chars..."

echo "Offending files and lines:"
(( bad_lines = 0 ))
(( inside_block = 0 ))
for file in "$@" ; do
  echo "$file"
  (( line_no = 0 ))
  while IFS="" read -r line || [[ -n "$line" ]] ; do
    (( line_no++ ))
    if [[ "$line" =~ ^'```' ]] ; then
      (( inside_block = !$inside_block ))
      continue
    fi
    if ! (( $inside_block )) \
        && ! [[ "$line" =~ " | "|"-|-"|"://"|"]:"|\[\^[^\ ]+\]: ]] \
        && (( "${#line}" > $MAX_LINE_LENGTH )) ; then
      (( bad_lines++ ))
      echo -e "\t$line_no : $line"
    fi
  done < "$file"
done

echo "$bad_lines offending lines found."
(( $bad_lines == 0 ))
