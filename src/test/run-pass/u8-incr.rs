// -*- rust -*-

fn main() {
  let u8 x = 12 as u8;
  let u8 y = 12 as u8;
  x = x + (1 as u8);
  x = x - (1 as u8);
  check(x == y);
  //x = 14 as u8;
  //x = x + 1 as u8;
}

