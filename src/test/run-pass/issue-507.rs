
/*
   This is a test case for Issue 507.

   https://github.com/graydon/rust/issues/507
*/

use std;

import task;
import task::join;
import comm;
import comm::chan;
import comm::send;
import comm::port;
import comm::recv;

fn grandchild(c: chan<int>) { send(c, 42); }

fn child(c: chan<int>) {
    let _grandchild = task::spawn_joinable(copy c, grandchild);
    join(_grandchild);
}

fn main() {
    let p = comm::port();

    let _child = task::spawn_joinable(chan(p), child);

    let x: int = recv(p);

    log(debug, x);

    assert (x == 42);

    join(_child);
}
