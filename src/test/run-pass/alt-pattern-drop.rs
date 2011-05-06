// -*- rust -*-

use std;
import std.Str;

// FIXME: import std.Dbg.const_refcount. Currently
// cross-crate const references don't work.
const uint const_refcount = 0x7bad_face_u;

tag t {
  make_t(str);
  clam;
}

fn foo(str s) {
  let t x = make_t(s);  // ref up

  alt (x) {
    case (make_t(?y)) { log y; }  // ref up then down
    case (_) { log "?"; fail; }
  }

  log Str.refcount(s);
  assert (Str.refcount(s) == const_refcount);
}

fn main() {
  let str s = "hi";     // ref up
  foo(s);               // ref up then down
  log Str.refcount(s);
  assert (Str.refcount(s) == const_refcount);
}
