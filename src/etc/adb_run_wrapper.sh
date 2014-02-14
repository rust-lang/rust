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
