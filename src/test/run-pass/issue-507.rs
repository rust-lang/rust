
/*
   This is a test case for Issue 507.

   https://github.com/graydon/rust/issues/507
*/

use std;

import task;
import comm;
import comm::chan;
import comm::send;
import comm::port;
import comm::recv;

fn grandchild(c: chan<int>) { send(c, 42); }

fn child(c: chan<int>) {
    task::spawn(|| grandchild(c) )
}

fn main() {
    let p = comm::port();
    let ch = chan(p);

    task::spawn(|| child(ch) );

    let x: int = recv(p);

    log(debug, x);

    assert (x == 42);
}
