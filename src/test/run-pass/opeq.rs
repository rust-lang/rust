// -*- rust -*-

fn main() {
  let int x = 1;

  x *= 2;
  log x;
  check (x == 2);

  x += 3;
  log x;
  check (x == 5);

  x *= x;
  log x;
  check (x == 25);

  x /= 5;
  log x;
  check (x == 5);
}

