native "rust" mod rustrt {
    fn task_sleep(uint time_in_us);
    fn task_yield();
    fn task_join(task t) -> int;
    fn unsupervise();
    fn pin_task();
    fn unpin_task();
    fn clone_chan(*rust_chan c) -> *rust_chan;

    type rust_chan;
}

/**
 * Hints the scheduler to yield this task for a specified ammount of time.
 *
 * arg: time_in_us maximum number of microseconds to yield control for
 */
fn sleep(uint time_in_us) {
    ret rustrt::task_sleep(time_in_us);
}

fn yield() {
    ret rustrt::task_yield();
}

tag task_result {
    tr_success;
    tr_failure;
}

fn join(task t) -> task_result {
    alt (rustrt::task_join(t)) {
        0 { tr_success }
        _ { tr_failure }
    }
}

fn unsupervise() {
    ret rustrt::unsupervise();
}

fn pin() {
    rustrt::pin_task();
}

fn unpin() {
    rustrt::unpin_task();
}

fn clone_chan[T](chan[T] c) -> chan[T] {
    auto cloned = rustrt::clone_chan(unsafe::reinterpret_cast(c));
    ret unsafe::reinterpret_cast(cloned);
}

// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
