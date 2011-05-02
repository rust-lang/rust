// -*- rust -*-

fn main() {
  let int x = 15;
  let int y = 5;
  check(x / 5 == 3);
  check(x / 4 == 3);
  check(x / 3 == 5);
  check(x / y == 3);
  check(15 / y == 3);

  check(x % 5 == 0);
  check(x % 4 == 3);
  check(x % 3 == 0);
  check(x % y == 0);
  check(15 % y == 0);
}
