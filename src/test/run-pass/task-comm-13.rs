use std;
import std::task;

fn start(c: chan[int], start: int, number_of_messages: int) {
    let i: int = 0;
    while i < number_of_messages { c <| start + i; i += 1; }
}

fn main() {
    log "Check that we don't deadlock.";
    let p: port[int] = port();
    let a: task = spawn start(chan(p), 0, 10);
    task::join(a);
    log "Joined task";
}