#
# usage : adb_run_wrapper [test dir - where test executables exist] [test executable]
#

# Sometimes android shell produce exitcode "1 : Text File Busy"
# Retry after $WAIT seconds, expecting resource cleaned-up
WAIT=10
TEST_PATH=$1
BIN_PATH=/system/bin
if [ -d "$TEST_PATH" ]
then
    shift
    RUN=$1

    if [ ! -z "$RUN" ]
    then
        shift

        L_RET=1
        L_COUNT=0
        cd $TEST_PATH
        while [ $L_RET -eq 1 ]
        do
            TEST_EXEC_ENV=22 LD_LIBRARY_PATH=$TEST_PATH PATH=$BIN_PATH:$TEST_PATH $TEST_PATH/$RUN $@ 1>$TEST_PATH/$RUN.stdout 2>$TEST_PATH/$RUN.stderr
            L_RET=$?
            if [ $L_COUNT -gt 0 ]
            then
               /system/bin/sleep $WAIT
               /system/bin/sync
            fi
            L_COUNT=$((L_COUNT+1))
        done

        echo $L_RET > $TEST_PATH/$RUN.exitcode

    fi
fi
