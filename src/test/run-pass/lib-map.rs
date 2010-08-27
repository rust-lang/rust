// -*- rust -*-

use std;
import std.map;
import std._str;
import std.util;

fn test_simple() {
  log "*** starting test_simple";

  fn eq_uint(&uint x, &uint y) -> bool { ret x == y; }
  fn hash_uint(&uint u) -> uint {
    // FIXME: can't use std.util.id since we'd be capturing a type param,
    // and presently we can't close items over type params.
    ret u;
  }

  let map.hashfn[uint] hasher_uint = hash_uint;
  let map.eqfn[uint] eqer_uint = eq_uint;

  let map.hashfn[str] hasher_str = _str.hash;
  let map.eqfn[str] eqer_str = _str.eq;


  log "uint -> uint";

  let map.hashmap[uint, uint] hm_uu = map.mk_hashmap[uint, uint](hasher_uint,
                                                                 eqer_uint);

  check (hm_uu.insert(10u, 12u));
  check (hm_uu.insert(11u, 13u));
  check (hm_uu.insert(12u, 14u));

  check (hm_uu.get(11u) == 13u);
  check (hm_uu.get(12u) == 14u);
  check (hm_uu.get(10u) == 12u);

  check (!hm_uu.insert(12u, 14u));
  check (hm_uu.get(12u) == 14u);

  check (!hm_uu.insert(12u, 12u));
  check (hm_uu.get(12u) == 12u);


  /*
  log "str -> uint";

  let map.hashmap[str, uint] hm_su = map.mk_hashmap[str, uint](hasher_str,
                                                               eqer_str);

  check (hm_su.insert("ten", 12u));
  check (hm_su.insert("eleven", 13u));
  check (hm_su.insert("twelve", 14u));

  check (hm_su.get("eleven") == 13u);
  check (hm_su.get("twelve") == 14u);
  check (hm_su.get("ten") == 12u);

  check (!hm_su.insert("twelve", 14u));
  check (hm_su.get("twelve") == 14u);

  check (!hm_su.insert("twelve", 12u));
  check (hm_su.get("twelve") == 12u);


  log "uint -> str";

  let map.hashmap[uint, str] hm_us = map.mk_hashmap[uint, str](hasher_uint,
                                                               eqer_uint);

  check (hm_us.insert(10u, "twelve"));
  check (hm_us.insert(11u, "thirteen"));
  check (hm_us.insert(12u, "fourteen"));

  check (_str.eq(hm_us.get(11u), "thirteen"));
  check (_str.eq(hm_us.get(12u), "fourteen"));
  check (_str.eq(hm_us.get(10u), "twelve"));

  check (!hm_us.insert(12u, "fourteen"));
  check (_str.eq(hm_us.get(12u), "fourteen"));

  check (!hm_us.insert(12u, "twelve"));
  check (_str.eq(hm_us.get(12u), "twelve"));


  log "str -> str";

  let map.hashmap[str, str] hm_ss = map.mk_hashmap[str, str](hasher_str,
                                                             eqer_str);

  check (hm_ss.insert("ten", "twelve"));
  check (hm_ss.insert("eleven", "thirteen"));
  check (hm_ss.insert("twelve", "fourteen"));

  check (_str.eq(hm_ss.get("eleven"), "thirteen"));
  check (_str.eq(hm_ss.get("twelve"), "fourteen"));
  check (_str.eq(hm_ss.get("ten"), "twelve"));

  check (!hm_ss.insert("twelve", "fourteen"));
  check (_str.eq(hm_ss.get("twelve"), "fourteen"));

  check (!hm_ss.insert("twelve", "twelve"));
  check (_str.eq(hm_ss.get("twelve"), "twelve"));
  */


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
    check (hm.insert(i, i * i));
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

  check (hm.insert(num_to_insert, 17u));
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

fn test_removal() {
  log "*** starting test_removal";

  let uint num_to_insert = 64u;

  fn eq(&uint x, &uint y) -> bool { ret x == y; }
  fn hash(&uint u) -> uint {
    // This hash function intentionally causes collisions between
    // consecutive integer pairs.
    ret (u / 2u) * 2u;
  }

  check (hash(0u) == hash(1u));
  check (hash(2u) == hash(3u));
  check (hash(0u) != hash(2u));

  let map.hashfn[uint] hasher = hash;
  let map.eqfn[uint] eqer = eq;
  let map.hashmap[uint, uint] hm = map.mk_hashmap[uint, uint](hasher, eqer);

  let uint i = 0u;
  while (i < num_to_insert) {
    check (hm.insert(i, i * i));
    log "inserting " + std._uint.to_str(i, 10u)
      + " -> " + std._uint.to_str(i * i, 10u);
    i += 1u;
  }

  check (hm.size() == num_to_insert);

  log "-----";
  log "removing evens";

  i = 0u;
  while (i < num_to_insert) {
    /**
     * FIXME (issue #150): we want to check the removed value as in the
     * following:

    let util.option[uint] v = hm.remove(i);
    alt (v) {
      case (util.some[uint](u)) {
        check (u == (i * i));
      }
      case (util.none[uint]()) { fail; }
    }

     * but we util.option is a tag type so util.some and util.none are
     * off limits until we parse the dwarf for tag types.
     */

    hm.remove(i);
    i += 2u;
  }

  check (hm.size() == (num_to_insert / 2u));

  log "-----";

  i = 1u;
  while (i < num_to_insert) {
    log "get(" + std._uint.to_str(i, 10u) + ") = "
      + std._uint.to_str(hm.get(i), 10u);
    check (hm.get(i) == i * i);
    i += 2u;
  }

  log "-----";
  log "rehashing";

  hm.rehash();

  log "-----";

  i = 1u;
  while (i < num_to_insert) {
    log "get(" + std._uint.to_str(i, 10u) + ") = "
      + std._uint.to_str(hm.get(i), 10u);
    check (hm.get(i) == i * i);
    i += 2u;
  }

  log "-----";

  i = 0u;
  while (i < num_to_insert) {
    check (hm.insert(i, i * i));
    log "inserting " + std._uint.to_str(i, 10u)
      + " -> " + std._uint.to_str(i * i, 10u);
    i += 2u;
  }

  check (hm.size() == num_to_insert);

  log "-----";

  i = 0u;
  while (i < num_to_insert) {
    log "get(" + std._uint.to_str(i, 10u) + ") = "
      + std._uint.to_str(hm.get(i), 10u);
    check (hm.get(i) == i * i);
    i += 1u;
  }

  log "-----";
  log "rehashing";

  hm.rehash();

  log "-----";

  check (hm.size() == num_to_insert);

  i = 0u;
  while (i < num_to_insert) {
    log "get(" + std._uint.to_str(i, 10u) + ") = "
      + std._uint.to_str(hm.get(i), 10u);
    check (hm.get(i) == i * i);
    i += 1u;
  }

  log "*** finished test_removal";
}

fn main() {
  test_simple();
  test_growth();
  test_removal();
}
