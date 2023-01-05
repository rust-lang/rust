#![allow(overflowing_literals)]
#![deny(unreachable_patterns)]

fn test(val: u8) {
  match val {
    256 => print!("0b1110\n"),
    512 => print!("0b1111\n"),
    //~^ ERROR: unreachable pattern
    _   => print!("fail\n"),
  }
}

fn main() {
  test(1);
}
