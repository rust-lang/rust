// xfail-stage3
use std;
import std::comm;

fn start(pcc: *u8) {
    let c = comm::chan_from_unsafe_ptr(pcc);
    let p : comm::_port[int] = comm::mk_port();
    c.send(p.mk_chan().unsafe_ptr());
}

fn main() {
    let p = comm::mk_port();
    let child = spawn start(p.mk_chan().unsafe_ptr());
    let pc = p.recv();
    let c : comm::_chan[int] = comm::chan_from_unsafe_ptr(pc);
}
