// -*- rust -*-

use std;
import std::comm;
import std::comm::send;
import std::comm::_chan;
import std::task;

fn a(c: _chan<int>) {
    if true {
        log "task a";
        log "task a";
        log "task a";
        log "task a";
        log "task a";
    }
    send(c, 10);
}

fn k(x: int) -> int { ret 15; }

fn g(x: int, y: str) -> int { log x; log y; let z: int = k(1); ret z; }

fn main() {
    let n: int = 2 + 3 * 7;
    let s: str = "hello there";
    let p = comm::mk_port();
    task::_spawn(bind a(p.mk_chan()));
    task::_spawn(bind b(p.mk_chan()));
    let x: int = 10;
    x = g(n, s);
    log x;
    n = p.recv();
    n = p.recv();
    // FIXME: use signal-channel for this.
    log "children finished, root finishing";
}

fn b(c: _chan<int>) {
    if true {
        log "task b";
        log "task b";
        log "task b";
        log "task b";
        log "task b";
        log "task b";
    }
    send(c, 10);
}
