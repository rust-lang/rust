// -*- rust -*-

use std;
import std::task;
import std::comm::*;

fn main() {
    let p = port();
    let y: int;

    task::spawn(chan(p), child);
    y = recv(p);
    log "received 1";
    log y;
    assert (y == 10);

    task::spawn(chan(p), child);
    y = recv(p);
    log "received 2";
    log y;
    assert (y == 10);
}

fn# child(c: chan<int>) { send(c, 10); }
