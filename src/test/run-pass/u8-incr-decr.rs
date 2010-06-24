// -*- rust -*-

// These constants were chosen because they aren't used anywhere
// in the rest of the generated code so they're easily grep-able.

fn main() {
  let u8 x = u8(19); // 0x13
  let u8 y = u8(35); // 0x23
  x = x + u8(7);     // 0x7
  y = y - u8(9);     // 0x9
  check(x == y);
}
