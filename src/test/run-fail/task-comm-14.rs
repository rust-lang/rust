// xfail-stage0
// xfail-stage1
// xfail-stage2
// xfail-stage3
io fn main() {
    let port[int] po = port();

    // Spawn 10 tasks each sending us back one int.
    let int i = 10;
    while (i > 0) {
        log i;
        spawn "child" child(i, chan(po));
        i = i - 1;
    }

    // Spawned tasks are likely killed before they get a chance to send
    // anything back, so we deadlock here.

    i = 10;
    let int value = 0;
    while (i > 0) {
        log i;
        po |> value;
        i = i - 1;
    }

    log "main thread exiting";
}

io fn child(int x, chan[int] ch) {
    log x;
    ch <| x;
}
