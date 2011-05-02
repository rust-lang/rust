// -*- rust -*-

type point = rec(int x, int y, int z);

fn f(&point p) {
  check (p.z == 12);
}

fn main() {
  let point x = rec(x=10, y=11, z=12);
  f(x);
}
