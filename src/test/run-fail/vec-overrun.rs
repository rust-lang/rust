// xfail-stage0
// xfail-stage1
// xfail-stage2
// -*- rust -*-

// error-pattern:bounds check

fn main() {
  let vec[int] v = vec(10);
  let int x = 0;
  assert (v.(x) == 10);
  // Bounds-check failure.
  assert (v.(x + 2) == 20);
}
