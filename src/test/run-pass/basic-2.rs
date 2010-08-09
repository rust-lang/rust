// -*- rust -*-

io fn a(chan[int] c) {
    log "task a0";
    log "task a1";
    c <| 10;
}

io fn main() {
    let port[int] p = port();
    spawn a(chan(p));
    spawn b(chan(p));
    let int n = 0;
    n <- p;
    n <- p;
    log "Finished.";
}

io fn b(chan[int] c) {
    log "task b0";
    log "task b1";
    log "task b2";
    log "task b2";
    log "task b3";
    c <| 10;
}
