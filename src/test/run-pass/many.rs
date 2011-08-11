// -*- rust -*-

fn sub(parent: chan[int], id: int) {
  if (id == 0) {
    parent <| 0;
  } else {
    let p: port[int] = port();
    let child = spawn sub(chan(p), id-1);
    let y: int; p |> y;
    parent <| y + 1;
  }
}

fn main() {
  let p: port[int] = port();
  let child = spawn sub(chan(p), 200);
  let y: int; p |> y;
  log "transmission complete";
  log y;
  assert (y == 200);
}
