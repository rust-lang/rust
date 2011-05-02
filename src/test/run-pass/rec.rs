// -*- rust -*-

type rect = rec(int x, int y, int w, int h);

fn f(rect r, int x, int y, int w, int h) {
  check (r.x == x);
  check (r.y == y);
  check (r.w == w);
  check (r.h == h);
}

fn main() {
  let rect r = rec(x=10, y=20, w=100, h=200);
  check (r.x == 10);
  check (r.y == 20);
  check (r.w == 100);
  check (r.h == 200);
  let rect r2 = r;
  let int x = r2.x;
  check (x == 10);
  f(r, 10, 20, 100, 200);
  f(r2, 10, 20, 100, 200);
}
