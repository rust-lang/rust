// xfail-stage0
// -*- rust -*-

fn a(chan[int] c) {
  if (true) {
    log "task a";
    log "task a";
    log "task a";
    log "task a";
    log "task a";
  }
  c <| 10;
}

fn k(int x) -> int {
  ret 15;
}

fn g(int x, str y) -> int {
  log x;
  log y;
  let int z = k(1);
  ret z;
}

fn main() {
    let int n = 2 + 3 * 7;
    let str s = "hello there";
    let port[int] p = port();
    spawn a(chan(p));
    spawn b(chan(p));
    let int x = 10;
    x = g(n,s);
    log x;
    p |> n;
    p |> n;
    // FIXME: use signal-channel for this.
    log "children finished, root finishing";
}

fn b(chan[int] c) {
  if (true) {
    log "task b";
    log "task b";
    log "task b";
    log "task b";
    log "task b";
    log "task b";
  }
  c <| 10;
}
