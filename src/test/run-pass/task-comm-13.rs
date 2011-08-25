use std;
import std::task;
import std::comm;
import std::comm::send;

fn start(c: comm::chan<int>, start: int, number_of_messages: int) {
    let i: int = 0;
    while i < number_of_messages { send(c, start + i); i += 1; }
}

fn main() {
    log "Check that we don't deadlock.";
    let p = comm::port();
    let a = task::spawn_joinable(bind start(comm::chan(p), 0, 10));
    task::join(a);
    log "Joined task";
}
