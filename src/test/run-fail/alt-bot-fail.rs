// xfail-stage0
// xfail-stage1
// xfail-stage2
// xfail-stage3
use std;
import std::option::*;

fn foo(str s) {
}

fn main() {
  auto i = alt (some[int](3)) {
    case (none[int]) {
      fail
    }
    case (some[int](_)) {
      fail
    }
  };
  foo(i);
}