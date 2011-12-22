// xfail-test
// -*- rust -*-

use std (name = "std",
         url = "http://rust-lang.org/src/std",
         uuid = _, ver = _);

fn main() {
  auto s = str.alloc(10 as uint);
  s += "hello ";
  log_full(core::debug, s);
  s += "there";
  log_full(core::debug, s);
  auto z = vec.alloc::<int>(10 as uint);
}
