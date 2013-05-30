#
# usage : adb_run_wrapper [test dir - where test executables exist] [test executable]
#
PATH=$1
if [ -d "$PATH" ]
then
    shift
    RUN=$1

    if [ ! -z "$RUN" ]
    then
        shift
        while [ -f $PATH/lock ]
        do
            /system/bin/sleep 1
        done
        /system/bin/touch $PATH/lock
        LD_LIBRARY_PATH=$PATH $PATH/$RUN $@ 1>$PATH/$RUN.stdout 2>$PATH/$RUN.stderr
        echo $? > $PATH/$RUN.exitcode
        /system/bin/rm $PATH/lock
        /system/bin/sync
    fi
fi
