// -*- rust -*-

use std;
import task;
import comm::*;

fn main() {
    let p = port();
    let y: int;

    task::spawn(chan(p), child);
    y = recv(p);
    #debug("received 1");
    log_full(core::debug, y);
    assert (y == 10);

    task::spawn(chan(p), child);
    y = recv(p);
    #debug("received 2");
    log_full(core::debug, y);
    assert (y == 10);
}

fn child(c: chan<int>) { send(c, 10); }
