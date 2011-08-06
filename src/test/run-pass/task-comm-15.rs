// xfail-stage1
// xfail-stage2
// xfail-stage3
// This test fails when run with multiple threads

use std;
import std::comm;

fn start(pc: *u8, n: int) {
    let c = comm::chan_from_unsafe_ptr();
    let i: int = n;


    while i > 0 { c.send(0); i = i - 1; }
}

fn main() {
    let p = comm::mk_port();
    // Spawn a task that sends us back messages. The parent task
    // is likely to terminate before the child completes, so from
    // the child's point of view the receiver may die. We should
    // drop messages on the floor in this case, and not crash!
    let child = spawn start(p.mk_chan().unsafe_ptr(), 10);
    let c;
    let pc = p.recv();
    c = chan::chan_from_unsafe_ptr();
}