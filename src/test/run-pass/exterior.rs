// -*- rust -*-

type point = rec(int x, int y, mutable int z);

fn f(@point p) {
  check (p.z == 12);
  p.z = 13;
  check (p.z == 13);
}

fn main() {
  let point a = rec(x=10, y=11, z=mutable 12);
  let @point b = a;
  check (b.z == 12);
  f(b);
  check (a.z == 12);
  check (b.z == 13);
}
