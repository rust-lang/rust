// -*- rust -*-

use std;
import std._str;

type t = tag(make_t(str), clam());

fn main() {
  let str s = "hi";     // ref up
  let t x = make_t(s);  // ref up

  alt (x) {
    case (make_t(y)) { log y; }  // ref up and ref down
    case (_) { log "?"; fail; }
  }

  log _str.refcount(s);
  check (_str.refcount(s) == 2u);
}
