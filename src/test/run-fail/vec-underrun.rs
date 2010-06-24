// -*- rust -*-

// error-pattern:bounds check

fn main() {
  let vec[int] v = vec(10, 20);
  let int x = 0;
  check (v.(x) == 10);
  // Bounds-check failure.
  check (v.(x-1) == 20);
}
