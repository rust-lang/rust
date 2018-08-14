#!/bin/bash -x

# Copyright 2018 The Rust Project Developers. See the COPYRIGHT
# file at the top-level directory of this distribution and at
# http://rust-lang.org/COPYRIGHT.
#
# Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
# http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
# <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
# option. This file may not be copied, modified, or distributed
# except according to those terms.

# Usage: $0 project_name url sha1
# Get the crate with the specified sha1.
#
# all arguments are required.
#
# See below link for git usage:
# https://stackoverflow.com/questions/3489173#14091182

# Mandatory arguments:
PROJECT_NAME=$1
URL=$2
SHA1=$3

function err_exit() {
    echo "ERROR:" $*
    exit 1
}

git clone $URL $PROJECT_NAME || err_exit
cd $PROJECT_NAME || err_exit
git reset --hard $SHA1 || err_exit
