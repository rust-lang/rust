// xfail-test
// This checks that preemption works.

fn starve_main(alive: chan<int>) {
    log "signalling main";
    alive <| 1;
    log "starving main";
    let i: int = 0;
    while true { i += 1; }
}

fn main() {
    let alive: port<int> = port();
    log "main started";
    let s: task = spawn starve_main(chan(alive));
    let i: int;
    log "main waiting for alive signal";
    alive |> i;
    log "main got alive signal";
    while i < 50 { log "main iterated"; i += 1; }
    log "main completed";
}
