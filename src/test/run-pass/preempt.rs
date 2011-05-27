// xfail-stage0
// xfail-stage1
// xfail-stage2
// This checks that preemption works.

fn starve_main(chan[int] alive) {
  log "signalling main";
  alive <| 1;
  log "starving main";
  let int i = 0;
  while (true) {
    i += 1;
  }
}

fn main() {
  let port[int] alive = port();
  log "main started";
  let task s = spawn starve_main(chan(alive));
  let int i;
  log "main waiting for alive signal";
  alive |> i;
  log "main got alive signal";
  while (i < 50) {
    log "main iterated";
    i += 1;
  }
  log "main completed";
}