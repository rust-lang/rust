// -*- rust -*-

use std;
import std.map;

fn test_simple() {
  log "*** starting test_simple";

  fn eq(&uint x, &uint y) -> bool { ret x == y; }
  fn hash(&uint u) -> uint {
    // FIXME: can't use std.util.id since we'd be capturing a type param,
    // and presently we can't close items over type params.
    ret u;
  }

  let map.hashfn[uint] hasher = hash;
  let map.eqfn[uint] eqer = eq;
  let map.hashmap[uint, uint] hm = map.mk_hashmap[uint, uint](hasher, eqer);
  hm.insert(10u, 12u);
  hm.insert(11u, 13u);
  hm.insert(12u, 14u);

  check (hm.get(11u) == 13u);
  check (hm.get(12u) == 14u);
  check (hm.get(10u) == 12u);

  log "*** finished test_simple";
}

fn main() {
  test_simple();
}
