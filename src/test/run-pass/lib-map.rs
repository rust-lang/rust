// -*- rust -*-

use std;
import std.Map;
import std.Str;
import std.UInt;
import std.Util;

fn test_simple() {
  log "*** starting test_simple";

  fn eq_uint(&uint x, &uint y) -> bool { ret x == y; }
  fn hash_uint(&uint u) -> uint {
    // FIXME: can't use std.Util.id since we'd be capturing a type param,
    // and presently we can't close items over type params.
    ret u;
  }

  let Map.hashfn[uint] hasher_uint = hash_uint;
  let Map.eqfn[uint] eqer_uint = eq_uint;

  let Map.hashfn[str] hasher_str = Str.hash;
  let Map.eqfn[str] eqer_str = Str.eq;


  log "uint -> uint";

  let Map.hashmap[uint, uint] hm_uu = Map.mk_hashmap[uint, uint](hasher_uint,
                                                                 eqer_uint);

  assert (hm_uu.insert(10u, 12u));
  assert (hm_uu.insert(11u, 13u));
  assert (hm_uu.insert(12u, 14u));

  assert (hm_uu.get(11u) == 13u);
  assert (hm_uu.get(12u) == 14u);
  assert (hm_uu.get(10u) == 12u);

  assert (!hm_uu.insert(12u, 14u));
  assert (hm_uu.get(12u) == 14u);

  assert (!hm_uu.insert(12u, 12u));
  assert (hm_uu.get(12u) == 12u);


  let str ten = "ten";
  let str eleven = "eleven";
  let str twelve = "twelve";

  log "str -> uint";

  let Map.hashmap[str, uint] hm_su = Map.mk_hashmap[str, uint](hasher_str,
                                                               eqer_str);
  assert (hm_su.insert("ten", 12u));
  assert (hm_su.insert(eleven, 13u));
  assert (hm_su.insert("twelve", 14u));

  assert (hm_su.get(eleven) == 13u);

  assert (hm_su.get("eleven") == 13u);
  assert (hm_su.get("twelve") == 14u);
  assert (hm_su.get("ten") == 12u);

  assert (!hm_su.insert("twelve", 14u));
  assert (hm_su.get("twelve") == 14u);

  assert (!hm_su.insert("twelve", 12u));
  assert (hm_su.get("twelve") == 12u);


  log "uint -> str";

  let Map.hashmap[uint, str] hm_us = Map.mk_hashmap[uint, str](hasher_uint,
                                                               eqer_uint);

  assert (hm_us.insert(10u, "twelve"));
  assert (hm_us.insert(11u, "thirteen"));
  assert (hm_us.insert(12u, "fourteen"));

  assert (Str.eq(hm_us.get(11u), "thirteen"));
  assert (Str.eq(hm_us.get(12u), "fourteen"));
  assert (Str.eq(hm_us.get(10u), "twelve"));

  assert (!hm_us.insert(12u, "fourteen"));
  assert (Str.eq(hm_us.get(12u), "fourteen"));

  assert (!hm_us.insert(12u, "twelve"));
  assert (Str.eq(hm_us.get(12u), "twelve"));


  log "str -> str";

  let Map.hashmap[str, str] hm_ss = Map.mk_hashmap[str, str](hasher_str,
                                                             eqer_str);

  assert (hm_ss.insert(ten, "twelve"));
  assert (hm_ss.insert(eleven, "thirteen"));
  assert (hm_ss.insert(twelve, "fourteen"));

  assert (Str.eq(hm_ss.get("eleven"), "thirteen"));
  assert (Str.eq(hm_ss.get("twelve"), "fourteen"));
  assert (Str.eq(hm_ss.get("ten"), "twelve"));

  assert (!hm_ss.insert("twelve", "fourteen"));
  assert (Str.eq(hm_ss.get("twelve"), "fourteen"));

  assert (!hm_ss.insert("twelve", "twelve"));
  assert (Str.eq(hm_ss.get("twelve"), "twelve"));

  log "*** finished test_simple";
}

/**
 * Force map growth and rehashing.
 */
fn test_growth() {
  log "*** starting test_growth";

  let uint num_to_insert = 64u;

  fn eq_uint(&uint x, &uint y) -> bool { ret x == y; }
  fn hash_uint(&uint u) -> uint {
    // FIXME: can't use std.Util.id since we'd be capturing a type param,
    // and presently we can't close items over type params.
    ret u;
  }


  log "uint -> uint";

  let Map.hashfn[uint] hasher_uint = hash_uint;
  let Map.eqfn[uint] eqer_uint = eq_uint;
  let Map.hashmap[uint, uint] hm_uu = Map.mk_hashmap[uint, uint](hasher_uint,
                                                                 eqer_uint);

  let uint i = 0u;
  while (i < num_to_insert) {
    assert (hm_uu.insert(i, i * i));
    log "inserting " + UInt.to_str(i, 10u)
      + " -> " + UInt.to_str(i * i, 10u);
    i += 1u;
  }

  log "-----";

  i = 0u;
  while (i < num_to_insert) {
    log "get(" + UInt.to_str(i, 10u) + ") = "
      + UInt.to_str(hm_uu.get(i), 10u);
    assert (hm_uu.get(i) == i * i);
    i += 1u;
  }

  assert (hm_uu.insert(num_to_insert, 17u));
  assert (hm_uu.get(num_to_insert) == 17u);

  log "-----";

  hm_uu.rehash();

  i = 0u;
  while (i < num_to_insert) {
    log "get(" + UInt.to_str(i, 10u) + ") = "
      + UInt.to_str(hm_uu.get(i), 10u);
    assert (hm_uu.get(i) == i * i);
    i += 1u;
  }


  log "str -> str";

  let Map.hashfn[str] hasher_str = Str.hash;
  let Map.eqfn[str] eqer_str = Str.eq;
  let Map.hashmap[str, str] hm_ss = Map.mk_hashmap[str, str](hasher_str,
                                                             eqer_str);

  i = 0u;
  while (i < num_to_insert) {
    assert (hm_ss.insert(UInt.to_str(i, 2u), UInt.to_str(i * i, 2u)));
    log "inserting \"" + UInt.to_str(i, 2u)
      + "\" -> \"" + UInt.to_str(i * i, 2u) + "\"";
    i += 1u;
  }

  log "-----";

  i = 0u;
  while (i < num_to_insert) {
    log "get(\""
      + UInt.to_str(i, 2u)
      + "\") = \""
      + hm_ss.get(UInt.to_str(i, 2u)) + "\"";

    assert (Str.eq(hm_ss.get(UInt.to_str(i, 2u)),
                   UInt.to_str(i * i, 2u)));
    i += 1u;
  }

  assert (hm_ss.insert(UInt.to_str(num_to_insert, 2u),
                      UInt.to_str(17u, 2u)));

  assert (Str.eq(hm_ss.get(UInt.to_str(num_to_insert, 2u)),
                 UInt.to_str(17u, 2u)));

  log "-----";

  hm_ss.rehash();

  i = 0u;
  while (i < num_to_insert) {
    log "get(\"" + UInt.to_str(i, 2u) + "\") = \""
      + hm_ss.get(UInt.to_str(i, 2u)) + "\"";
    assert (Str.eq(hm_ss.get(UInt.to_str(i, 2u)),
                   UInt.to_str(i * i, 2u)));
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

  assert (hash(0u) == hash(1u));
  assert (hash(2u) == hash(3u));
  assert (hash(0u) != hash(2u));

  let Map.hashfn[uint] hasher = hash;
  let Map.eqfn[uint] eqer = eq;
  let Map.hashmap[uint, uint] hm = Map.mk_hashmap[uint, uint](hasher, eqer);

  let uint i = 0u;
  while (i < num_to_insert) {
    assert (hm.insert(i, i * i));
    log "inserting " + UInt.to_str(i, 10u)
      + " -> " + UInt.to_str(i * i, 10u);
    i += 1u;
  }

  assert (hm.size() == num_to_insert);

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
        assert (u == (i * i));
      }
      case (util.none[uint]()) { fail; }
    }

     * but we util.option is a tag type so util.some and util.none are
     * off limits until we parse the dwarf for tag types.
     */

    hm.remove(i);
    i += 2u;
  }

  assert (hm.size() == (num_to_insert / 2u));

  log "-----";

  i = 1u;
  while (i < num_to_insert) {
    log "get(" + UInt.to_str(i, 10u) + ") = "
      + UInt.to_str(hm.get(i), 10u);
    assert (hm.get(i) == i * i);
    i += 2u;
  }

  log "-----";
  log "rehashing";

  hm.rehash();

  log "-----";

  i = 1u;
  while (i < num_to_insert) {
    log "get(" + UInt.to_str(i, 10u) + ") = "
      + UInt.to_str(hm.get(i), 10u);
    assert (hm.get(i) == i * i);
    i += 2u;
  }

  log "-----";

  i = 0u;
  while (i < num_to_insert) {
    assert (hm.insert(i, i * i));
    log "inserting " + UInt.to_str(i, 10u)
      + " -> " + UInt.to_str(i * i, 10u);
    i += 2u;
  }

  assert (hm.size() == num_to_insert);

  log "-----";

  i = 0u;
  while (i < num_to_insert) {
    log "get(" + UInt.to_str(i, 10u) + ") = "
      + UInt.to_str(hm.get(i), 10u);
    assert (hm.get(i) == i * i);
    i += 1u;
  }

  log "-----";
  log "rehashing";

  hm.rehash();

  log "-----";

  assert (hm.size() == num_to_insert);

  i = 0u;
  while (i < num_to_insert) {
    log "get(" + UInt.to_str(i, 10u) + ") = "
      + UInt.to_str(hm.get(i), 10u);
    assert (hm.get(i) == i * i);
    i += 1u;
  }

  log "*** finished test_removal";
}

fn main() {
  test_simple();
  test_growth();
  test_removal();
}
