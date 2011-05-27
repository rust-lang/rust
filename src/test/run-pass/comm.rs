// xfail-stage0
// xfail-stage1
// xfail-stage2
// -*- rust -*-

fn main() {
  let port[int] p = port();
  spawn child(chan(p));
  let int y;
  p |> y;
  log "received";
  log y;
  assert (y == 10);
}

fn child(chan[int] c) {
  c <| 10;
}

