/**
 * At the moment, this is a partial hashmap implementation, not yet fit for
 * use, but useful as a stress test for rustboot.
 */

import std._int;
import std.sys;
import std.util;
import std._vec;


type hashfn[K] = fn(K) -> uint;
type eqfn[K] = fn(K) -> bool;

type hashmap[K, V] = obj {
  fn insert(&K key, &V val);
  fn contains_key(&K key) -> bool;
  fn get(&K key) -> V;
  fn find(&K key) -> util.option[V];
  fn remove(&K key) -> util.option[V];
  fn rehash();
};

fn mk_hashmap[K, V](&hashfn[K] hasher, &eqfn[K] eqer) -> hashmap[K, V] {

  let uint initial_capacity = uint(32); // 2^5
  let util.rational load_factor = rec(num=3, den=4);

  type bucket[V] = tag(nil(), deleted(), some(V));

  // Derive two hash functions from the one given by taking the upper
  // half and lower half of the uint bits.  Our bucket probing
  // sequence is then defined by
  //
  //   hash(key, i) := hashl(key) + i * hashr(key)   for i = 0, 1, 2, ...
  //
  // Tearing the hash function apart this way is kosher in practice
  // as, assuming 32-bit uints, the table would have to be at 2^32
  // buckets before the resulting pair of hash functions no longer
  // probes all buckets for a fixed key.  Note that hashr is made to
  // output odd numbers (hence coprime to the number of nbkts, which
  // is always a power of 2), so that all buckets are probed for a
  // fixed key.

  fn hashl[K](hashfn[K] hasher, uint nbkts, &K key) -> uint {
    ret (hasher(key) >>> (sys.rustrt.size_of[uint]() * uint(8) / uint(2)))
      % nbkts;
  }

  fn hashr[K](hashfn[K] hasher, uint nbkts, &K key) -> uint {
    ret ((((~ uint(0)) >>> (sys.rustrt.size_of[uint]() * uint(8) / uint(2)))
          & hasher(key)) * uint(2) + uint(1))
      % nbkts;
  }

  fn hash[K](hashfn[K] hasher, uint nbkts, &K key, uint i) -> uint {
    ret hashl[K](hasher, nbkts, key) + i * hashr[K](hasher, nbkts, key);
  }

  fn find_common[K, V](hashfn[K] hasher,
                       vec[mutable bucket[V]] bkts,
                       uint nbkts,
                       &K key)
    -> util.option[V]
  {
    let uint i = uint(0);
    while (i < nbkts) {
      // Pending fix to issue #94, remove uint coercion.
      let int j = int(hash[K](hasher, nbkts, key, i));
      alt (bkts.(j)) {
        case (some[V](val)) {
          ret util.some[V](val);
        }
        case (nil[V]()) {
          ret util.none[V]();
        }
        case (deleted[V]()) {
          i += uint(1);
        }
      }
    }
    ret util.none[V]();
  }

  obj hashmap[K, V](hashfn[K] hasher,
                    eqfn[K] eqer,
                    mutable vec[mutable bucket[V]] bkts,
                    mutable uint nbkts,
                    mutable uint nelts,
                    util.rational lf)
  {
    fn insert(&K key, &V val) {
      // FIXME grow the table and rehash if we ought to.
      let uint i = uint(0);
      while (i < nbkts) {
        // Issue #94, as in find_common()
        let int j = int(hash[K](hasher, nbkts, key, i));
        alt (bkts.(j)) {
          case (some[V](_)) {
            i += uint(1);
          }
          case (_) {
            bkts.(j) = some[V](val);
            nelts += uint(1);
            ret;
          }
        }
      }
      // full table, impossible unless growth is broken. remove after testing.
      fail;
    }

    fn contains_key(&K key) -> bool {
      alt (find_common[K, V](hasher, bkts, nbkts, key)) {
        case (util.some[V](_)) { ret true; }
        case (_) { ret false; }
      }
    }

    fn get(&K key) -> V {
      alt (find_common[K, V](hasher, bkts, nbkts, key)) {
        case (util.some[V](val)) { ret val; }
        case (_) { fail; }
      }
    }

    fn find(&K key) -> util.option[V] {
      be find_common[K, V](hasher, bkts, nbkts, key);
    }

    fn remove(&K key) -> util.option[V] {
      let uint i = uint(0);
      while (i < nbkts) {
        // Issue #94, as in find_common()
        let int j = int(hash[K](hasher, nbkts, key, i));
        alt (bkts.(j)) {
          case (some[V](val)) {
            bkts.(j) = deleted[V]();
            ret util.some[V](val);
          }
          case (deleted[V]()) {
            nelts += uint(1);
          }
          case (nil[V]()) {
            ret util.none[V]();
          }
        }
      }
      ret util.none[V]();
    }

    fn rehash() {}
  }

  let vec[mutable bucket[V]] bkts =
    _vec.init_elt[mutable bucket[V]](nil[V](), initial_capacity);

  ret hashmap[K, V](hasher, eqer, bkts, uint(0), uint(0), load_factor);
}
