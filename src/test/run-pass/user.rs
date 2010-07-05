// -*- rust -*-

use std (name = "std",
         url = "http://rust-lang.org/src/std",
         uuid = _, ver = _);

fn main() {
  auto s = std._str.alloc(uint(10));
  s += "hello ";
  log s;
  s += "there";
  log s;
  auto z = std._vec.alloc[int](uint(10));
}
