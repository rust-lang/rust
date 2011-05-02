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
      check (c == ('h' as u8));
    }
    if (i == 1) {
      check (c == ('e' as u8));
    }
    if (i == 2) {
      check (c == ('l' as u8));
    }
    if (i == 3) {
      check (c == ('l' as u8));
    }
    if (i == 4) {
      check (c == ('o' as u8));
    }
    // ...
    i += 1;
    log i;
    log c;
  }
  check(i == 11);
}
