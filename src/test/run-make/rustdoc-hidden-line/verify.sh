#!/bin/sh

file="$1/doc/foo/fn.foo.html"

grep -v 'invisible' $file &&
grep '#.*\[.*derive.*(.*Eq.*).*\].*//.*Bar' $file

exit $?
