// -*- rust -*-

use std;
import std::comm;
import std::comm::send;
import std::comm::_chan;
import std::task;

fn a(c: _chan<int>) { log "task a0"; log "task a1"; send(c, 10); }

fn main() {
    let p = comm::mk_port();
    task::_spawn(bind a(p.mk_chan()));
    task::_spawn(bind b(p.mk_chan()));
    let n: int = 0;
    n = p.recv();
    n = p.recv();
    log "Finished.";
}

fn b(c: _chan<int>) {
    log "task b0";
    log "task b1";
    log "task b2";
    log "task b2";
    log "task b3";
    send(c, 10);
}
