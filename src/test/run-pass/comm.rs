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
    log_err "received";
    log_err y;
    assert (y == 10);
}

fn child(c: chan<int>) {
    log_err "sending";
    send(c, 10);
    log_err "value sent"
}
