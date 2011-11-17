// xfail-win32
// Create a task that is supervised by another task,
// join the supervised task from the supervising task,
// then fail the supervised task. The supervised task
// will kill the supervising task, waking it up. The
// supervising task no longer needs to be wakened when
// the supervised task exits.

use std;
import std::task;

fn supervised(&&_args: ()) {
    // Yield to make sure the supervisor joins before we
    // fail. This is currently not needed because the supervisor
    // runs first, but I can imagine that changing.
    task::yield();
    fail;
}

fn supervisor(&&_args: ()) {
    // Unsupervise this task so the process doesn't return a failure status as
    // a result of the main task being killed.
    task::unsupervise();
    let f = supervised;
    let t = task::spawn_joinable((), supervised);
    task::join(t);
}

fn main() {
    let dom2 = task::spawn_joinable((), supervisor);
    task::join(dom2);
}

// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
