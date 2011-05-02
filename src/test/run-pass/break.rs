// xfail-boot

fn main() {
  auto i = 0;
  while (i < 20) {
    i += 1;
    if (i == 10) { break; }
  }
  check(i == 10);

  do {
    i += 1;
    if (i == 20) { break; }
  } while (i < 30);
  check(i == 20);

  for (int x in vec(1, 2, 3, 4, 5, 6)) {
    if (x == 3) { break; }
    check(x <= 3);
  }

  i = 0;
  while (i < 10) {
    i += 1;
    if (i % 2 == 0) { cont; }
    check(i % 2 != 0);
  }

  i = 0;
  do {
    i += 1;
    if (i % 2 == 0) { cont; }
    check(i % 2 != 0);
  } while (i < 10);

  for (int x in vec(1, 2, 3, 4, 5, 6)) {
    if (x % 2 == 0) { cont; }
    check(x % 2 != 0);
  }
}
