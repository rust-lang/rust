#!/bin/sh

# Copyright 2014 The Rust Project Developers. See the COPYRIGHT
# file at the top-level directory of this distribution and at
# http://rust-lang.org/COPYRIGHT.
#
# Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
# http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
# <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
# option. This file may not be copied, modified, or distributed
# except according to those terms.

# This script, invoked e.g. "add-authors.sh 1.0.0..rust-lang/master",
# will merge new authors into AUTHORS.txt, obeying the mailmap
# file.
#
# After running this script, run `git diff` to manually inspect
# changes. If there are incorrect additions fix it by editing
# .mailmap and re-running the script.

if [ "${1-}" = "" ]; then
    echo "Usage: add-authors.sh 1.0.0..rust-lang/master"
    exit 1
fi

set -u -e

range="$1"

authors_file="./AUTHORS.txt"
tmp_file="./AUTHORS.txt.tmp"
old_authors="$(cat "$authors_file" | tail -n +2 | sed "/^$/d" | sort)"
new_authors="$(git log "$range" --format="%aN <%aE>" | sort | uniq)"

printf "%s\n\n" "Rust was written by these fine people:" > "$tmp_file"
printf "%s\n%s" "$old_authors" "$new_authors" | sort | uniq >> "$tmp_file"
mv -f "$tmp_file" "$authors_file"
