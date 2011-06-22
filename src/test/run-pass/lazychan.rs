// xfail-stage0
// -*- rust -*-

fn main() {
  let port[int] p = port();
  auto c = chan(p);
  let int y;

  spawn child(c);
  p |> y;
  log "received 1";
  log y;
  assert (y == 10);

  spawn child(c);
  p |> y;
  log "received 2";
  log y;
  assert (y == 10);
}

fn child(chan[int] c) {
  c <| 10;
}
