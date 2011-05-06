// xfail-stage0
// xfail-stage1
// xfail-stage2
// -*- rust -*-

use std (name = "std",
         url = "http://rust-lang.org/src/std",
         uuid = _, ver = _);

fn main() {
  auto s = std.Str.alloc(10 as uint);
  s += "hello ";
  log s;
  s += "there";
  log s;
  auto z = std.Vec.alloc[int](10 as uint);
}
