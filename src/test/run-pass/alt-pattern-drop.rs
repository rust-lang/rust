// -*- rust -*-

use std;
import std._str;

type t = tag(make_t(str), clam());

fn foo(str s) {
  let t x = make_t(s);  // ref up

  alt (x) {
    case (make_t(y)) { log y; }  // ref up then down
    case (_) { log "?"; fail; }
  }

  log _str.refcount(s);
  check (_str.refcount(s) == 3u);
}

fn main() {
  let str s = "hi";     // ref up
  foo(s);               // ref up then down
  log _str.refcount(s);
  check (_str.refcount(s) == 1u);
}
