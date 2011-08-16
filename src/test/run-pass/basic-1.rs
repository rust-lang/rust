// -*- rust -*-

use std;
import std::comm::_chan;
import std::comm::mk_port;
import std::comm::send;
import std::task;

fn a(c: _chan<int>) { send(c, 10); }

fn main() {
    let p = mk_port();
    task::_spawn(bind a(p.mk_chan()));
    task::_spawn(bind b(p.mk_chan()));
    let n: int = 0;
    n = p.recv();
    n = p.recv();
    //    log "Finished.";
}

fn b(c: _chan<int>) {
    //    log "task b0";
    //    log "task b1";
    //    log "task b2";
    //    log "task b3";
    //    log "task b4";
    //    log "task b5";
    send(c, 10);
}
