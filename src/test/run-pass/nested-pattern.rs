// a bug was causing this to complain about leaked memory on exit

use std;
import std.Option;
import std.Option.some;
import std.Option.none;

tag t {
  foo(int, uint);
  bar(int, Option.t[int]);
}

fn nested(t o) {

  alt (o) {
    case (bar(?i, some[int](_))) {
      log_err "wrong pattern matched";
      fail;
    }
    case (_) {
      log_err "succeeded";
    }
  }

}

fn main() {
  nested (bar (1, none[int]));
}