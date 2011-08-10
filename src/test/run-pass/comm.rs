// -*- rust -*-

use std;
import std::comm;
import std::comm::_chan;
import std::comm::send;
import std::task;

fn main() {
    let p = comm::mk_port();
    let t = task::_spawn(bind child(p.mk_chan()));
    let y = p.recv();
    log_err "received";
    log_err y;
    assert (y == 10);
}

fn child(c: _chan<int>) {
    log_err "sending";
    send(c, 10);
    log_err "value sent"
}
