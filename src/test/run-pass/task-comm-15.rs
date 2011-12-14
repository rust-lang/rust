// xfail-win32
use std;
import comm;
import task;

fn start(&&args: (comm::chan<int>, int)) {
    let (c, i) = args;

    while i > 0 { comm::send(c, 0); i = i - 1; }
}

fn main() {
    let p = comm::port();
    // Spawn a task that sends us back messages. The parent task
    // is likely to terminate before the child completes, so from
    // the child's point of view the receiver may die. We should
    // drop messages on the floor in this case, and not crash!
    let child = task::spawn((comm::chan(p), 10), start);
    let c = comm::recv(p);
}
