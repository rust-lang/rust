// Jump back and forth between the OS main thread and a new scheduler.
// The OS main scheduler should continue to be available and not terminate
// while it is not in use.

fn main() {
    run(100);
}

fn run(i: int) {

    log(debug, i);

    if i == 0 {
        ret;
    }

    do task::task().sched_mode(task::osmain).unlinked().spawn {
        task::yield();
        do task::task().sched_mode(task::single_threaded).unlinked().spawn {
            task::yield();
            run(i - 1);
            task::yield();
        }
        task::yield();
    }
}
