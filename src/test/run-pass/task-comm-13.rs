// xfail-stage0

use std;
import std::task;

fn start(chan[int] c, int start, int number_of_messages) {
    let int i = 0;
    while (i < number_of_messages) {
        c <| start + i;
        i += 1;
    }    
}

fn main() -> () {
    log "Check that we don't deadlock.";
    let port[int] p = port();
    let task a = spawn start(chan(p), 0, 10);
    task::join(a);
    log "Joined task";
}