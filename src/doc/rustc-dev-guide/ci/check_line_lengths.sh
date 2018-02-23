#!/bin/bash

echo "Checking line lengths in all source files <= $MAX_LINE_LENGTH chars..."

echo "Offending files and lines:"
(( success = 1 ))
for file in "$@" ; do
  echo "$file"
  (( line_no = 0 ))
  while IFS="" read -r line || [[ -n "$line" ]] ; do
    (( line_no++ ))
    if (( "${#line}" > $MAX_LINE_LENGTH )) ; then
      (( success = 0 ))
      echo -e "\t$line_no : $line"
    fi
  done < "$file"
done

(( $success )) && echo "No offending lines found."
