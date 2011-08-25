// -*- rust -*-

use std;
import std::comm;
import std::comm::send;
import std::comm::chan;
import std::comm::recv;
import std::task;

fn a(c: chan<int>) {
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
    let p = comm::port();
    task::spawn(bind a(chan(p)));
    task::spawn(bind b(chan(p)));
    let x: int = 10;
    x = g(n, s);
    log x;
    n = recv(p);
    n = recv(p);
    // FIXME: use signal-channel for this.
    log "children finished, root finishing";
}

fn b(c: chan<int>) {
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
