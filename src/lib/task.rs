native "rust" mod rustrt {
    fn task_sleep(time_in_us: uint);
    fn task_yield();
    fn task_join(t: task) -> int;
    fn unsupervise();
    fn pin_task();
    fn unpin_task();
    fn clone_chan(c: *rust_chan) -> *rust_chan;

    type rust_chan;

    fn set_min_stack(stack_size: uint);
}

/**
 * Hints the scheduler to yield this task for a specified ammount of time.
 *
 * arg: time_in_us maximum number of microseconds to yield control for
 */
fn sleep(time_in_us: uint) { ret rustrt::task_sleep(time_in_us); }

fn yield() { ret rustrt::task_yield(); }

tag task_result { tr_success; tr_failure; }

fn join(t: task) -> task_result {
    alt rustrt::task_join(t) { 0 { tr_success } _ { tr_failure } }
}

fn unsupervise() { ret rustrt::unsupervise(); }

fn pin() { rustrt::pin_task(); }

fn unpin() { rustrt::unpin_task(); }

fn clone_chan[T](c: chan[T]) -> chan[T] {
    let cloned = rustrt::clone_chan(unsafe::reinterpret_cast(c));
    ret unsafe::reinterpret_cast(cloned);
}

fn send[T](c: chan[T], v: &T) { c <| v; }

fn recv[T](p: port[T]) -> T { let v; p |> v; v }

fn set_min_stack(uint stack_size) {
    rustrt::set_min_stack(stack_size);
}

// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
