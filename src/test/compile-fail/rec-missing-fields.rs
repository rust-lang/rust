// -*- rust -*-

// error-pattern: mismatched types

// Issue #51.

type point = rec(int x, int y);

fn main() {
  let point p = rec(x=10);
  log p.y;
}
