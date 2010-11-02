// -*- rust -*-

type point = rec(int x, int y, mutable int z);

impure fn f(& mutable point p) {
  p.z = 13;
}

impure fn main() {
  let point x = rec(x=10, y=11, mutable z=12);
  f(x);
  check (x.z == 13);
}
