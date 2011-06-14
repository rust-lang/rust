// -*- rust -*-
use std;
import std::option;
import std::option::some;

// error-pattern: mismatched types

tag bar {
  t1((), option::t[vec[int]]);
  t2;
}

fn foo(bar t) {
  alt (t) {
    case (t1(_, some[int](?x))) {
      log x;
    }
    case (_) {
      fail;
    }
  }
}

fn main() {
}