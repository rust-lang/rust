// -*- rust -*-

use std;
import std::option;
import std::option::some;

// error-pattern: mismatched types

tag bar {
  t1((), option::t[vec[int]]);
  t2;
}

fn foo(bar t) -> int {
  alt (t) {
    case (t1(_, some(?x))) {
      ret (x * 3);
    }
    case (_) {
      fail;
    }
  }
}

fn main() {
}