use std;
import std::task;
import std::comm;
import std::comm::send;

fn start(&&args: (comm::chan<int>, int, int)) {
    let (c, start, number_of_messages) = args;
    let i: int = 0;
    while i < number_of_messages { send(c, start + i); i += 1; }
}

fn main() {
    log "Check that we don't deadlock.";
    let p = comm::port::<int>();
    let a = task::spawn_joinable((comm::chan(p), 0, 10), start);
    task::join(a);
    log "Joined task";
}
