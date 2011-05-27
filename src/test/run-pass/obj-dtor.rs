// xfail-stage0
// xfail-stage1
// xfail-stage2
obj worker(chan[int] c) {
  drop {
    log "in dtor";
    c <| 10;
  }
}

fn do_work(chan[int] c) {
  log "in child task";
  {
    let worker w = worker(c);
    log "constructed worker";
  }
  log "destructed worker";
  while(true) {
    // Deadlock-condition not handled properly yet, need to avoid
    // exiting the child early.
    c <| 11;
    yield;
  }
}

fn main() {
  let port[int] p = port();
  log "spawning worker";
  auto w = spawn do_work(chan(p));
  let int i;
  log "parent waiting for shutdown";
  p |> i;
  log "received int";
  assert (i == 10);
  log "int is OK, child-dtor ran as expected";
}