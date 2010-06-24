// -*- rust -*-

fn main() {
  let i8 x = i8(-12);
  let i8 y = i8(-12);
  x = x + i8(1);
  x = x - i8(1);
  check(x == y);
}
