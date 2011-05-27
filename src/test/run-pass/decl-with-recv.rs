// -*- rust -*-

fn main() {
  let port[int] po = port();
  let chan[int] ch = chan(po);

  ch <| 10;
  let int i; po |> i;
  assert (i == 10);

  ch <| 11;
  auto j; po |> j;
  assert (j == 11);
}
