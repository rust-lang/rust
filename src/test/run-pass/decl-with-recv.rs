// -*- rust -*-

fn main() {
  let port[int] po = port();
  let chan[int] ch = chan(po);

  ch <| 10;
  let int i <- po;
  check (i == 10);

  ch <| 11;
  auto j <- po;
  check (j == 11);
}
