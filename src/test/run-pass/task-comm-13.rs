use std;
import std::task;
import std::comm;

fn start(pc: *u8, start: int, number_of_messages: int) {
    let c = comm::chan_from_unsafe_ptr(pc);
    let i: int = 0;
    while i < number_of_messages { c.send(start + i); i += 1; }
}

fn main() {
    log "Check that we don't deadlock.";
    let p : comm::_port[int] = comm::mk_port();
    let a: task = spawn start(p.mk_chan().unsafe_ptr(), 0, 10);
    task::join(a);
    log "Joined task";
}