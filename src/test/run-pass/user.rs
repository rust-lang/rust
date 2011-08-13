// xfail-stage1
// xfail-stage2
// xfail-stage3
// -*- rust -*-

use std (name = "std",
         url = "http://rust-lang.org/src/std",
         uuid = _, ver = _);

fn main() {
  auto s = std::str.alloc(10 as uint);
  s += "hello ";
  log s;
  s += "there";
  log s;
  auto z = std::vec.alloc::<int>(10 as uint);
}
