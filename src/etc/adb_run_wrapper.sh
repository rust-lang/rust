# Copyright 2014 The Rust Project Developers. See the COPYRIGHT
# file at the top-level directory of this distribution and at
# http://rust-lang.org/COPYRIGHT.
#
# Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
# http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
# <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
# option. This file may not be copied, modified, or distributed
# except according to those terms.
#
# ignore-tidy-linelength
#
# usage : adb_run_wrapper [test dir - where test executables exist] [test executable]
#

TEST_PATH=$1
BIN_PATH=/system/bin
if [ -d "$TEST_PATH" ]
then
    shift
    RUN=$1

    if [ ! -z "$RUN" ]
    then
        shift

        cd $TEST_PATH
        TEST_EXEC_ENV=22 LD_LIBRARY_PATH=$TEST_PATH PATH=$BIN_PATH:$TEST_PATH $TEST_PATH/$RUN $@ 1>$TEST_PATH/$RUN.stdout 2>$TEST_PATH/$RUN.stderr
        L_RET=$?

        echo $L_RET > $TEST_PATH/$RUN.exitcode

    fi
fi
