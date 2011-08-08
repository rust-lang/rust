use std;
import std::task;
import std::comm;

fn start(pcc: *u8) {
    let c = comm::chan_from_unsafe_ptr(pcc);
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
    let p = comm::mk_port[*u8]();
    let child = spawn start(p.mk_chan().unsafe_ptr());

    let pc = p.recv();
    let c = comm::chan_from_unsafe_ptr(pc);
    c.send("A");
    c.send("B");
    task::yield();
}
