// -*- rust -*-

// error-pattern: record

type point = rec(int x, int y);

fn main() {
  let point origin = rec(x=0, y=0);

  let point origin3d = rec(z=0 with origin);
}
