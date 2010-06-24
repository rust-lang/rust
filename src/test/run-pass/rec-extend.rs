// -*- rust -*-

type point = rec(int x, int y);

fn main() {
  let point origin = rec(x=0, y=0);

  let point right = rec(x=origin.x + 10 with origin);
  let point up = rec(y=origin.y + 10 with origin);

  check(origin.x == 0);
  check(origin.y == 0);

  check(right.x == 10);
  check(right.y == 0);

  check(up.x == 0);
  check(up.y == 10);
}
