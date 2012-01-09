// xfail-test

// A port of task-killjoin to use a resource to manage
// the join.

use std;
import task;

fn joinable(f: fn()) -> (task::task, comm::port<bool>) {
    resource notify(data: (comm::chan<bool>,
                           @mutable bool)) {
        let (c, v) = data;
        #error["notify: task=%d v=%x unwinding=%b b=%b",
               task::get_task(),
               ptr::addr_of(*v) as uint,
               task::currently_unwinding(),
               *v];
        comm::send(c, *v);
    }
    fn wrapper(pair: (comm::chan<bool>, fn())) {
        let (c, f) = pair;
        let b = @mutable false;
        #error["wrapper: task=%d allocated v=%x",
               task::get_task(),
               ptr::addr_of(*b) as uint];
        let _r = notify((c, b));
        f();
        *b = true;
    }
    let p = comm::port();
    let c = comm::chan(p);
    let t = task::spawn {|| wrapper((c, f)) };
    ret (t, p);
}

fn join(pair: (task::task, comm::port<bool>)) -> bool {
    let (_, port) = pair;
    comm::recv(port)
}

fn supervised() {
    // Yield to make sure the supervisor joins before we
    // fail. This is currently not needed because the supervisor
    // runs first, but I can imagine that changing.
    #error["supervised task=%d", task::get_task()];
    task::yield();
    fail;
}

fn supervisor() {
    // Unsupervise this task so the process doesn't return a failure status as
    // a result of the main task being killed.
    task::unsupervise();
    #error["supervisor task=%d", task::get_task()];
    let t = joinable(supervised);
    join(t);
}

fn main() {
    join(joinable(supervisor));
}

// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
