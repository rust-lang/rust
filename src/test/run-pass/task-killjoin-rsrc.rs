// xfail-test

// A port of task-killjoin to use a resource to manage
// the join.

use std;
import task;

fn joinable(f: fn()) -> (task::task, comm::port<bool>) {
    resource notify(data: (comm::chan<bool>,
                           @mutable bool)) {
        let (c, v) = data;
        comm::send(c, *v);
    }
    fn wrapper(pair: (comm::chan<bool>, fn())) {
        let (c, f) = pair;
        let b = @mutable false;
        let _r = notify((c, b));
        f();
        *b = true;
    }
    let p = comm::port();
    let c = comm::chan(p);
    let t = task::spawn((c, f), wrapper);
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
    task::yield();
    fail;
}

fn supervisor() {
    // Unsupervise this task so the process doesn't return a failure status as
    // a result of the main task being killed.
    task::unsupervise();
    let f = supervised;
    join(joinable(supervised));
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
