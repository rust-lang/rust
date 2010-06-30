fn incr(& mutable int x) -> bool {
  x += 1;
  check (false);
  ret false;
}

fn main() {

  auto x = (1 == 2) || (3 == 3);
  check (x);

  let int y = 10;
  log x || incr(y);
  check (y == 10);

  if (true && x) {
    check (true);
  } else {
    check (false);
  }

}