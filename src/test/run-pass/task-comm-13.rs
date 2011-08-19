use std;
import std::task;
import std::comm;
import std::comm::send;

fn start(c: comm::_chan<int>, start: int, number_of_messages: int) {
    let i: int = 0;
    while i < number_of_messages { send(c, start + i); i += 1; }
}

fn main() {
    log "Check that we don't deadlock.";
    let p: comm::_port<int> = comm::mk_port();
    let a = task::_spawn(bind start(p.mk_chan(), 0, 10));
    task::join_id(a);
    log "Joined task";
}
