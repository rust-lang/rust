
PATH=$(echo $0 | sed 's#/[^/]*$##')
RUN=$1

if [ ! -z "$RUN" ]
then
    shift
    while [ -f $PATH/lock ]
    do
        sleep 1
    done
    touch $PATH/lock
	LD_LIBRARY_PATH=$PATH $PATH/$RUN $@ 1>$PATH/$RUN.stdout 2>$PATH/$RUN.stderr
	echo $? > $PATH/$RUN.exitcode
    /system/bin/rm $PATH/lock
fi

