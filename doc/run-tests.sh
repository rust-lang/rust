#!/bin/bash
rm -f fragments/*.rs
mkdir -p fragments
node extract-tests.js $1
for F in `ls fragments/*.rs`; do
  $RUSTC $F > /dev/null
  if [[ $? != 0 ]] ; then echo $F; fi
done
