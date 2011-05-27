// xfail-stage0
// xfail-stage1
// xfail-stage2
// -*- rust -*-

fn sub(chan[int] parent, int id) {
  if (id == 0) {
    parent <| 0;
  } else {
    let port[int] p = port();
    auto child = spawn sub(chan(p), id-1);
    let int y; p |> y;
    parent <| y + 1;
  }
}

fn main() {
  let port[int] p = port();
  auto child = spawn sub(chan(p), 500);
  let int p |> y;
  log "transmission complete";
  log y;
  assert (y == 500);
}
