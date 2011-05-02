tag foo {
  large;
  small;
}

fn main() {
  auto a = tup(1,2,3);
  auto b = tup(1,2,3);
  check (a == b);
  check (a != tup(1,2,4));
  check (a < tup(1,2,4));
  check (a <= tup(1,2,4));
  check (tup(1,2,4) > a);
  check (tup(1,2,4) >= a);
  auto x = large;
  auto y = small;
  check (x != y);
  check (x == large);
  check (x != small);
}