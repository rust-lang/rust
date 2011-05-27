// xfail-stage0
// xfail-stage1
// xfail-stage2
// -*- rust -*-

// error-pattern:1 == 2

fn child() {
  assert (1 == 2);
}

fn main() {
  let port[int] p = port();
  spawn child();
  let int x;
  p |> x;
}
