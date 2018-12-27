#!/bin/bash -x

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
