
/*
   This is a test case for Issue 507.

   https://github.com/graydon/rust/issues/507
*/

use std;

import std::task;
import std::task::join;
import std::comm;
import std::comm::chan;
import std::comm::send;
import std::comm::port;
import std::comm::recv;

fn# grandchild(c: chan<int>) { send(c, 42); }

fn# child(c: chan<int>) {
    let _grandchild = task::spawn_joinable2(c, grandchild);
    join(_grandchild);
}

fn main() {
    let p = comm::port();

    let _child = task::spawn_joinable2(chan(p), child);

    let x: int = recv(p);

    log x;

    assert (x == 42);

    join(_child);
}
