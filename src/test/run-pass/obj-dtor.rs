


// xfail-stage0
// xfail-stage1
// xfail-stage2
// xfail-stage3
obj worker(c: chan[int]) {drop { log "in dtor"; c <| 10; } }

fn do_work(c: chan[int]) {
    log "in child task";
    { let w: worker = worker(c); log "constructed worker"; }
    log "destructed worker";
    while true {
        // Deadlock-condition not handled properly yet, need to avoid
        // exiting the child early.

        c <| 11;
        yield;
    }
}

fn main() {
    let p: port[int] = port();
    log "spawning worker";
    let w = spawn do_work(chan(p));
    let i: int;
    log "parent waiting for shutdown";
    p |> i;
    log "received int";
    assert (i == 10);
    log "int is OK, child-dtor ran as expected";
}