use std;
import std::task;
import std::comm;

fn start(pcc: *u8) {
    let c = comm::chan_from_unsafe_ptr(pcc);
    let p;

    let a;
    let b;
    p = comm::mk_port[str]();
    c.send(p.mk_chan().unsafe_ptr());
    a = p.recv();
    log_err a;
    b = p.recv();
    log_err b;
}

fn main() {
    let p : comm::_port[*u8];
    let child;

    p = comm::mk_port();
    child = spawn start(p.mk_chan().unsafe_ptr());
    let pc; let c;

    pc = p.recv();
    c = comm::chan_from_unsafe_ptr(pc);
    c.send("A");
    c.send("B");
    task::yield();
}
