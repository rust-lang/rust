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

    let builder = task::builder();
    let opts = {
        sched: some({
            mode: task::osmain,
            foreign_stack_size: none
        })
        with task::get_opts(builder)
    };
    task::set_opts(builder, opts);
    task::unsupervise(builder);
    do task::run(builder) {
        task::yield();
        let builder = task::builder();
        let opts = {
            sched: some({
                mode: task::single_threaded,
                foreign_stack_size: none
            })
            with task::get_opts(builder)
        };
        task::set_opts(builder, opts);
        task::unsupervise(builder);
        do task::run(builder) {
            task::yield();
            run(i - 1);
            task::yield();
        }
        task::yield();
    }
}
