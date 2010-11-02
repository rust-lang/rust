// -*- rust -*-

impure fn main() {
  let port[int] p = port();
  spawn child(chan(p));
  let int y;
  y <- p;
  log "received";
  log y;
  check (y == 10);
}

impure fn child(chan[int] c) {
  c <| 10;
}

