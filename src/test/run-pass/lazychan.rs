// -*- rust -*-

use std;
import std::task;
import std::comm::*;

fn main() {
    let p = mk_port();
    let y: int;

    task::_spawn(bind child(p.mk_chan()));
    y = p.recv();
    log "received 1";
    log y;
    assert (y == 10);

    task::_spawn(bind child(p.mk_chan()));
    y = p.recv();
    log "received 2";
    log y;
    assert (y == 10);
}

fn child(c: _chan[int]) { send(c, 10); }