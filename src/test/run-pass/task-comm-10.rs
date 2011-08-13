// FIXME: this test is xfailed until sending strings is legal again.

//xfail-stage0
//xfail-stage1
//xfail-stage2
//xfail-stage3

use std;
import std::task;
import std::comm;

fn start(c: comm::_chan[str]) {
    let p = comm::mk_port[str]();
    c.send(p.mk_chan().unsafe_ptr());

    let a;
    let b;
    a = p.recv();
    log_err a;
    b = p.recv();
    log_err b;
}

fn main() {
    let p = comm::mk_port();
    let child = task::_spawn(bind start(p.mk_chan()));

    let c = p.recv();
    c.send("A");
    c.send("B");
    task::yield();
}
