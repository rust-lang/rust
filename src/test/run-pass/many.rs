// -*- rust -*-

io fn sub(chan[int] parent, int id) {
  if (id == 0) {
    parent <| 0;
  } else {
    let port[int] p = port();
    auto child = spawn sub(chan(p), id-1);
    let int y <- p;
    parent <| y + 1;
  }
}

io fn main() {
  let port[int] p = port();
  auto child = spawn sub(chan(p), 500);
  let int y <- p;
  log "transmission complete";
  log y;
  check (y == 500);
}
