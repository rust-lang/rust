// error-pattern:explicit failure
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
