#!/usr/bin/env bash

if [ "$1" == "--help" ]; then
  echo 'Usage:' "[MAX_LINE_LENGTH=n] $0 [file ...]"
  exit 1
fi

if [ "$MAX_LINE_LENGTH" == "" ]; then
    MAX_LINE_LENGTH=100
fi

if [ "$1" == "" ]; then
  shopt -s globstar
  files=( src/**/*.md )
else
  files=( "$@" )
fi

echo "Checking line lengths in all source files <= $MAX_LINE_LENGTH chars..."

echo "Offending files and lines:"
(( bad_lines = 0 ))
(( inside_block = 0 ))
for file in "${files[@]}"; do
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
