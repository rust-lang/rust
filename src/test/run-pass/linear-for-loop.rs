fn main() {
  auto x = vec(1,2,3);
  auto y = 0;
  for (int i in x) {
    log i;
    y += i;
  }
  log y;
  check (y == 6);

  auto s = "hello there";
  let int i = 0;
  for (u8 c in s) {
    if (i == 0) {
      check (c == u8('h'));
    }
    if (i == 1) {
      check (c == u8('e'));
    }
    if (i == 2) {
      check (c == u8('l'));
    }
    if (i == 3) {
      check (c == u8('l'));
    }
    if (i == 4) {
      check (c == u8('o'));
    }
    // ...
    if (i == 12) {
      check (c == u8(0));
    }
    i += 1;
    log i;
    log c;
  }
  check(i == 12);
}
