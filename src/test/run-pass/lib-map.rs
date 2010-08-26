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

  hm.insert(12u, 14u);
  check (hm.get(12u) == 14u);

  hm.insert(12u, 12u);
  check (hm.get(12u) == 12u);

  log "*** finished test_simple";
}

/**
 * Force map growth and rehashing.
 */
fn test_growth() {
  log "*** starting test_growth";

  let uint num_to_insert = 64u;

  fn eq(&uint x, &uint y) -> bool { ret x == y; }
  fn hash(&uint u) -> uint {
    // FIXME: can't use std.util.id since we'd be capturing a type param,
    // and presently we can't close items over type params.
    ret u;
  }

  let map.hashfn[uint] hasher = hash;
  let map.eqfn[uint] eqer = eq;
  let map.hashmap[uint, uint] hm = map.mk_hashmap[uint, uint](hasher, eqer);

  let uint i = 0u;
  while (i < num_to_insert) {
    hm.insert(i, i * i);
    log "inserting " + std._uint.to_str(i, 10u)
      + " -> " + std._uint.to_str(i * i, 10u);
    i += 1u;
  }

  log "-----";

  i = 0u;
  while (i < num_to_insert) {
    log "get(" + std._uint.to_str(i, 10u) + ") = "
      + std._uint.to_str(hm.get(i), 10u);
    check (hm.get(i) == i * i);
    i += 1u;
  }

  hm.insert(num_to_insert, 17u);
  check (hm.get(num_to_insert) == 17u);

  log "-----";

  hm.rehash();

  i = 0u;
  while (i < num_to_insert) {
    log "get(" + std._uint.to_str(i, 10u) + ") = "
      + std._uint.to_str(hm.get(i), 10u);
    check (hm.get(i) == i * i);
    i += 1u;
  }

  log "*** finished test_growth";
}

fn main() {
  test_simple();
  test_growth();
}
