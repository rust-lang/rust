// -*- rust -*-

use std;
import comm;
import comm::chan;
import comm::send;
import comm::recv;
import task;

fn main() {
    let p = comm::port();
    let t = task::spawn(chan(p), child);
    let y = recv(p);
    #error("received");
    log_full(core::error, y);
    assert (y == 10);
}

fn child(c: chan<int>) {
    #error("sending");
    send(c, 10);
    log_err "value sent"
}
