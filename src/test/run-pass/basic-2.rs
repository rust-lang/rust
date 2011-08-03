// -*- rust -*-

fn a(c: chan[int]) { log "task a0"; log "task a1"; c <| 10; }

fn main() {
    let p: port[int] = port();
    spawn a(chan(p));
    spawn b(chan(p));
    let n: int = 0;
    p |> n;
    p |> n;
    log "Finished.";
}

fn b(c: chan[int]) {
    log "task b0";
    log "task b1";
    log "task b2";
    log "task b2";
    log "task b3";
    c <| 10;
}