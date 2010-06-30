// -*- rust -*-

type point = rec(int x, int y, mutable int z);

fn f(& mutable point p) {
  p.z = 13;
}

fn main() {
  let point x = rec(x=10, y=11, z=mutable 12);
  f(x);
  check (x.z == 13);
}
