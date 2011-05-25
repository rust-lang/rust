// -*- rust -*-

// error-pattern:bounds check

fn main() {
  let vec[int] v = [10];
  let int x = 0;
  assert (v.(x) == 10);
  // Bounds-check failure.
  assert (v.(x + 2) == 20);
}
