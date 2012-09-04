
/*
   This is a test case for Issue 507.

   https://github.com/graydon/rust/issues/507
*/

use std;

import task;
import comm;
import comm::Chan;
import comm::send;
import comm::Port;
import comm::recv;

fn grandchild(c: Chan<int>) { send(c, 42); }

fn child(c: Chan<int>) {
    task::spawn(|| grandchild(c) )
}

fn main() {
    let p = comm::Port();
    let ch = Chan(p);

    task::spawn(|| child(ch) );

    let x: int = recv(p);

    log(debug, x);

    assert (x == 42);
}
