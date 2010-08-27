/**
 * At the moment, this is a partial hashmap implementation, not yet fit for
 * use, but useful as a stress test for rustboot.
 */

import std._int;
import std.sys;
import std.util;
import std._vec;


type hashfn[K] = fn(&K) -> uint;
type eqfn[K] = fn(&K, &K) -> bool;

type hashmap[K, V] = obj {
  fn size() -> uint;
  fn insert(&K key, &V val) -> bool;
  fn contains_key(&K key) -> bool;
  fn get(&K key) -> V;
  fn find(&K key) -> util.option[V];
  fn remove(&K key) -> util.option[V];
  fn rehash();
};

fn mk_hashmap[K, V](&hashfn[K] hasher, &eqfn[K] eqer) -> hashmap[K, V] {

  let uint initial_capacity = 32u; // 2^5
  let util.rational load_factor = rec(num=3, den=4);

  type bucket[K, V] = tag(nil(), deleted(), some(K, V));

  fn make_buckets[K, V](uint nbkts) -> vec[mutable bucket[K, V]] {
    ret _vec.init_elt[mutable bucket[K, V]](nil[K, V](), nbkts);
  }

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

  fn hashl[K](&hashfn[K] hasher, uint nbkts, &K key) -> uint {
    ret (hasher(key) >>> (sys.rustrt.size_of[uint]() * 8u / 2u));
  }

  fn hashr[K](&hashfn[K] hasher, uint nbkts, &K key) -> uint {
    ret ((((~ 0u) >>> (sys.rustrt.size_of[uint]() * 8u / 2u))
          & hasher(key)) * 2u + 1u);
  }

  fn hash[K](&hashfn[K] hasher, uint nbkts, &K key, uint i) -> uint {
    ret (hashl[K](hasher, nbkts, key)
         + i * hashr[K](hasher, nbkts, key)) % nbkts;
  }

  /**
   * We attempt to never call this with a full table.  If we do, it
   * will fail.
   */
  fn insert_common[K, V](&hashfn[K] hasher,
                         &eqfn[K] eqer,
                         vec[mutable bucket[K, V]] bkts,
                         uint nbkts,
                         &K key,
                         &V val)
    -> bool
  {
    let uint i = 0u;
    while (i < nbkts) {
      let uint j = hash[K](hasher, nbkts, key, i);
      alt (bkts.(j)) {
        case (some[K, V](k, _)) {
          if (eqer(key, k)) {
            bkts.(j) = some[K, V](k, val);
            ret false;
          }
          i += 1u;
        }
        case (_) {
          bkts.(j) = some[K, V](key, val);
          ret true;
        }
      }
    }
    fail; // full table
  }

  fn find_common[K, V](&hashfn[K] hasher,
                       &eqfn[K] eqer,
                       vec[mutable bucket[K, V]] bkts,
                       uint nbkts,
                       &K key)
    -> util.option[V]
  {
    let uint i = 0u;
    while (i < nbkts) {
      let uint j = (hash[K](hasher, nbkts, key, i));
      alt (bkts.(j)) {
        case (some[K, V](k, v)) {
          if (eqer(key, k)) {
            ret util.some[V](v);
          }
        }
        case (nil[K, V]()) {
          ret util.none[V]();
        }
        case (deleted[K, V]()) { }
      }
      i += 1u;
    }
    ret util.none[V]();
  }


  fn rehash[K, V](&hashfn[K] hasher,
                  &eqfn[K] eqer,
                  vec[mutable bucket[K, V]] oldbkts, uint noldbkts,
                  vec[mutable bucket[K, V]] newbkts, uint nnewbkts)
  {
    for (bucket[K, V] b in oldbkts) {
      alt (b) {
        case (some[K, V](k, v)) {
          insert_common[K, V](hasher, eqer, newbkts, nnewbkts, k, v);
        }
        case (_) { }
      }
    }
  }

  obj hashmap[K, V](hashfn[K] hasher,
                    eqfn[K] eqer,
                    mutable vec[mutable bucket[K, V]] bkts,
                    mutable uint nbkts,
                    mutable uint nelts,
                    util.rational lf)
  {
    fn size() -> uint { ret nelts; }

    fn insert(&K key, &V val) -> bool {
      let util.rational load = rec(num=(nelts + 1u) as int, den=nbkts as int);
      if (!util.rational_leq(load, lf)) {
        let uint nnewbkts = _uint.next_power_of_two(nbkts + 1u);
        let vec[mutable bucket[K, V]] newbkts = make_buckets[K, V](nnewbkts);
        rehash[K, V](hasher, eqer, bkts, nbkts, newbkts, nnewbkts);
        bkts = newbkts;
        nbkts = nnewbkts;
      }

      if (insert_common[K, V](hasher, eqer, bkts, nbkts, key, val)) {
        nelts += 1u;
        ret true;
      }
      ret false;
    }

    fn contains_key(&K key) -> bool {
      alt (find_common[K, V](hasher, eqer, bkts, nbkts, key)) {
        case (util.some[V](_)) { ret true; }
        case (_) { ret false; }
      }
    }

    fn get(&K key) -> V {
      alt (find_common[K, V](hasher, eqer, bkts, nbkts, key)) {
        case (util.some[V](val)) { ret val; }
        case (_) { fail; }
      }
    }

    fn find(&K key) -> util.option[V] {
      be find_common[K, V](hasher, eqer, bkts, nbkts, key);
    }

    fn remove(&K key) -> util.option[V] {
      let uint i = 0u;
      while (i < nbkts) {
        let uint j = (hash[K](hasher, nbkts, key, i));
        alt (bkts.(j)) {
          case (some[K, V](k, v)) {
            if (eqer(key, k)) {
              bkts.(j) = deleted[K, V]();
              nelts -= 1u;
              ret util.some[V](v);
            }
          }
          case (deleted[K, V]()) { }
          case (nil[K, V]()) {
            ret util.none[V]();
          }
        }
        i += 1u;
      }
      ret util.none[V]();
    }

    fn rehash() {
      let vec[mutable bucket[K, V]] newbkts = make_buckets[K, V](nbkts);
      rehash[K, V](hasher, eqer, bkts, nbkts, newbkts, nbkts);
      bkts = newbkts;
    }
  }

  let vec[mutable bucket[K, V]] bkts = make_buckets[K, V](initial_capacity);

  ret hashmap[K, V](hasher, eqer, bkts, initial_capacity, 0u, load_factor);
}
