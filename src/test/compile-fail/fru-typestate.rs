// -*- rust -*-

// error-pattern: precondition

type point = rec(int x, int y);

fn main() {
  let point origin;

  let point right = rec(x=10 with origin);
  origin = rec(x=0, y=0);
}
