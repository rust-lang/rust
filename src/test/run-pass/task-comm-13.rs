// xfail-stage0
// xfail-stage1
// xfail-stage2
use std;
import std::_task;

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
    let task a = spawn "start" start(chan(p), 0, 10);
    join a;
    log "Joined task";
}