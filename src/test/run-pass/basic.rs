// -*- rust -*-

fn a(c: chan[int]) {
    if true {
        log "task a";
        log "task a";
        log "task a";
        log "task a";
        log "task a";
    }
    c <| 10;
}

fn k(x: int) -> int { ret 15; }

fn g(x: int, y: str) -> int { log x; log y; let z: int = k(1); ret z; }

fn main() {
    let n: int = 2 + 3 * 7;
    let s: str = "hello there";
    let p: port[int] = port();
    spawn a(chan(p));
    spawn b(chan(p));
    let x: int = 10;
    x = g(n, s);
    log x;
    p |> n;
    p |> n;
    // FIXME: use signal-channel for this.
    log "children finished, root finishing";
}

fn b(c: chan[int]) {
    if true {
        log "task b";
        log "task b";
        log "task b";
        log "task b";
        log "task b";
        log "task b";
    }
    c <| 10;
}