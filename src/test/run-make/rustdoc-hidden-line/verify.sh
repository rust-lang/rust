#!/bin/sh

file="$1/doc/foo/fn.foo.html"

grep -v 'invisible' $file &&
grep '#.*\[.*deriving.*(.*Eq.*).*\].*//.*Bar' $file

exit $?
