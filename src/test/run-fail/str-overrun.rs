// -*- rust -*-

// error-pattern:bounds check

fn main() {
  let str s = "hello";
  let int x = 0;
  check (s.(x) == u8(0x68));

  // NB: at the moment a string always has a trailing NULL,
  // so the largest index value on the string above is 5, not
  // 4. Possibly change this.

  // Bounds-check failure.
  check (s.(x + 6) == u8(0x0));
}
