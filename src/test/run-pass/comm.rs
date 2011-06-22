// xfail-stage0
// -*- rust -*-

fn main() {
  let port[int] p = port();
  let task t = spawn child(chan(p));
  let int y;
  p |> y;
  log_err "received";
  log_err y;
  assert (y == 10);
}

fn child(chan[int] c) {
  log_err "sending";
  c <| 10;
  log_err "value sent"
}
