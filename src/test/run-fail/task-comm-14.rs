io fn main() {
    let port[int] po = port();

    // Spawn 10 tasks each sending us back one int.
    let int i = 10;
    while (i > 0) {
        log i;
        spawn "child" child(i, chan(po));
        i = i - 1;
    }

    i = 10;
    let int value = 0;
    while (i > 0) {
        log i;
        value <- po;
        i = i - 1;
    }
  
    log "main thread exiting";
}

io fn child(int x, chan[int] ch) {
    log x;
    ch <| x;
}
