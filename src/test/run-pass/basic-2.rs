// -*- rust -*-

use std;
import comm;
import comm::send;
import comm::chan;
import comm::recv;
import task;

fn a(c: chan<int>) { log "task a0"; log "task a1"; send(c, 10); }

fn main() {
    let p = comm::port();
    task::spawn(chan(p), a);
    task::spawn(chan(p), b);
    let n: int = 0;
    n = recv(p);
    n = recv(p);
    log "Finished.";
}

fn b(c: chan<int>) {
    log "task b0";
    log "task b1";
    log "task b2";
    log "task b2";
    log "task b3";
    send(c, 10);
}
