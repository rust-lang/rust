
/*
   This is a test case for Issue 507.

   https://github.com/graydon/rust/issues/507
*/

use std;

import std::task;
import std::task::join_id;
import std::comm;
import std::comm::_chan;
import std::comm::send;

fn grandchild(c: _chan[int]) { send(c, 42); }

fn child(c: _chan[int]) {
    let _grandchild = task::_spawn(bind grandchild(c));
    join_id(_grandchild);
}

fn main() {
    let p = comm::mk_port();

    let _child = task::_spawn(bind child(p.mk_chan()));

    let x: int = p.recv();

    log x;

    assert (x == 42);

    join_id(_child);
}