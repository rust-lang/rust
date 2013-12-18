#!/bin/sh

# $1 is the TMPDIR

dirs="doc doc/foo doc/foo/bar doc/foo/bar/baz doc/src doc/src/foo"

for dir in $dirs; do if [ ! -d $1/$dir ]; then
	echo "$1/$dir is not a directory!"
	exit 1
fi done

files="doc/foo/index.html doc/foo/bar/index.html doc/foo/bar/baz/fn.baz.html doc/foo/bar/trait.Doge.html doc/src/foo/foo.rs.html"

for file in $files; do if [ ! -f $1/$file ]; then
	echo "$1/$file is not a file!"
	exit 1
fi done
