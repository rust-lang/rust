// -*- rust -*-

fn main() {
  let u8 x = u8(12);
  let u8 y = u8(12);
  x = x + u8(1);
  x = x - u8(1);
  check(x == y);
  //x = u8(14);
  //x = x + u8(1);
}

