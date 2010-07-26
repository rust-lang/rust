// -*- rust -*-

// These constants were chosen because they aren't used anywhere
// in the rest of the generated code so they're easily grep-able.

fn main() {
  let u8 x = 19 as u8;  // 0x13
  let u8 y = 35 as u8;  // 0x23
  x = x + (7 as u8);    // 0x7
  y = y - (9 as u8);    // 0x9
  check(x == y);
}
