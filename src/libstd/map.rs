/*
Module: map

A hashmap
*/

/* Section: Types */

/*
Type: hashfn

A function that returns a hash of a value
*/
type hashfn<K> = fn(K) -> uint;

/*
Type: eqfn

Equality
*/
type eqfn<K> = fn(K, K) -> bool;

/*
Type: hashset

A convenience type to treat a hashmap as a set
*/
type hashset<K> = hashmap<K, ()>;

/*
Obj: hashmap
*/
type hashmap<K, V> = obj {
    /*
    Method: size

    Return the number of elements in the map
    */
    fn size() -> uint;
    /*
    Method: insert

    Add a value to the map. If the map already contains a value for
    the specified key then the original value is replaced.

    Returns:

    True if the key did not already exist in the map
    */
    fn insert(K, V) -> bool;
    /*
    Method: contains_key

    Returns true if the map contains a value for the specified key
    */
    fn contains_key(K) -> bool;
    /*
    Method: get

    Get the value for the specified key

    Failure:

    If the key does not exist in the map
    */
    fn get(K) -> V;
    /*
    Method: find

    Get the value for the specified key. If the key does not exist
    in the map then returns none.
    */
    fn find(K) -> option::t<V>;
    /*
    Method: remove

    Remove and return a value from the map. If the key does not exist
    in the map then returns none.
    */
    fn remove(K) -> option::t<V>;
    /*
    Method: rehash

    Force map growth and rehashing
    */
    fn rehash();
    /*
    Method: items

    Iterate over all the key/value pairs in the map
    */
    fn items(block(K, V));
    /*
    Method: keys

    Iterate over all the keys in the map
    */
    fn keys(block(K));
    /*
    Iterate over all the values in the map
    */
    fn values(block(V));
};

/* Section: Operations */

mod chained {
    type entry<copy K, copy V> = {
        hash: uint,
        key: K,
        mutable value: V,
        mutable next: chain<K, V>
    };

    tag chain<copy K, copy V> {
        present(@entry<K, V>);
        absent;
    }

    type t<copy K, copy V> = {
        mutable size: uint,
        mutable chains: [mutable chain<K,V>],
        hasher: hashfn<K>,
        eqer: eqfn<K>
    };

    tag search_result<copy K, copy V> {
        not_found;
        found_first(uint, @entry<K,V>);
        found_after(@entry<K,V>, @entry<K,V>);
    }

    fn search_rem<copy K, copy V>(tbl: t<K,V>,
                                  k: K,
                                  h: uint,
                                  idx: uint,
                                  e_root: @entry<K,V>) -> search_result<K,V> {
        let e0 = e_root;
        let comp = 1u;   // for logging
        while true {
            alt e0.next {
              absent. {
                log("search_tbl", "absent", "comparisons", comp,
                    "hash", h, "idx", idx);

                ret not_found;
              }
              present(e1) {
                comp += 1u;
                let e1_key = e1.key; // Satisfy alias checker.
                if e1.hash == h && tbl.eqer(e1_key, k) {
                    log("search_tbl", "present", "comparisons", comp,
                        "hash", h, "idx", idx);
                    ret found_after(e0, e1);
                } else {
                    e0 = e1;
                }
              }
            }
        }
        util::unreachable();
    }

    fn search_tbl<copy K, copy V>(
        tbl: t<K,V>, k: K, h: uint) -> search_result<K,V> {
        let idx = h % vec::len(tbl.chains);
        alt tbl.chains[idx] {
          absent. {
            log("search_tbl", "absent", "comparisons", 0u,
                "hash", h, "idx", idx);
            ret not_found;
          }
          present(e) {
            let e_key = e.key; // Satisfy alias checker.
            if e.hash == h && tbl.eqer(e_key, k) {
                log("search_tbl", "present", "comparisons", 1u,
                    "hash", h, "idx", idx);
                ret found_first(idx, e);
            } else {
                ret search_rem(tbl, k, h, idx, e);
            }
          }
        }
    }

    fn insert<copy K, copy V>(tbl: t<K,V>, k: K, v: V) -> bool {
        let hash = tbl.hasher(k);
        alt search_tbl(tbl, k, hash) {
          not_found. {
            tbl.size += 1u;
            let idx = hash % vec::len(tbl.chains);
            let old_chain = tbl.chains[idx];
            tbl.chains[idx] = present(@{
                hash: hash,
                key: k,
                mutable value: v,
                mutable next: old_chain});
            ret true;
          }
          found_first(_, entry) {
            entry.value = v;
            ret false;
          }
          found_after(_, entry) {
            entry.value = v;
            ret false
          }
        }
    }

    fn get<copy K, copy V>(tbl: t<K,V>, k: K) -> option::t<V> {
        alt search_tbl(tbl, k, tbl.hasher(k)) {
          not_found. {
            ret option::none;
          }

          found_first(_, entry) {
            ret option::some(entry.value);
          }

          found_after(_, entry) {
            ret option::some(entry.value);
          }
        }
    }

    fn remove<copy K, copy V>(tbl: t<K,V>, k: K) -> option::t<V> {
        alt search_tbl(tbl, k, tbl.hasher(k)) {
          not_found. {
            ret option::none;
          }

          found_first(idx, entry) {
            tbl.size -= 1u;
            tbl.chains[idx] = entry.next;
            ret option::some(entry.value);
          }

          found_after(eprev, entry) {
            tbl.size -= 1u;
            eprev.next = entry.next;
            ret option::some(entry.value);
          }
        }
    }

    fn chains<copy K, copy V>(nchains: uint) -> [mutable chain<K,V>] {
        ret vec::init_elt_mut(absent, nchains);
    }

    fn foreach_entry<copy K, copy V>(chain0: chain<K,V>,
                                     blk: block(@entry<K,V>)) {
        let chain = chain0;
        while true {
            alt chain {
              absent. { ret; }
              present(entry) {
                let next = entry.next;
                blk(entry); // may modify entry.next!
                chain = next;
              }
            }
        }
    }

    fn foreach_chain<copy K, copy V>(chains: [const chain<K,V>],
                                     blk: block(@entry<K,V>)) {
        let i = 0u, n = vec::len(chains);
        while i < n {
            foreach_entry(chains[i], blk);
            i += 1u;
        }
    }

    fn rehash<copy K, copy V>(tbl: t<K,V>) {
        let old_chains = tbl.chains;
        let n_old_chains = vec::len(old_chains);
        let n_new_chains: uint = uint::next_power_of_two(n_old_chains + 1u);
        tbl.chains = chains(n_new_chains);
        foreach_chain(old_chains) { |entry|
            let idx = entry.hash % n_new_chains;
            entry.next = tbl.chains[idx];
            tbl.chains[idx] = present(entry);
        }
    }

    fn items<copy K, copy V>(tbl: t<K,V>, blk: block(K,V)) {
        let tbl_chains = tbl.chains;  // Satisfy alias checker.
        foreach_chain(tbl_chains) { |entry|
            let key = entry.key;
            let value = entry.value;
            blk(key, value);
        }
    }

    obj o<copy K, copy V>(tbl: @t<K,V>,
                          lf: float) {
        fn size() -> uint {
            ret tbl.size;
        }

        fn insert(k: K, v: V) -> bool {
            let nchains = vec::len(tbl.chains);
            let load = (tbl.size + 1u as float) / (nchains as float);
            if load > lf {
                rehash(*tbl);
            }
            ret insert(*tbl, k, v);
        }

        fn contains_key(k: K) -> bool {
            ret option::is_some(get(*tbl, k));
        }

        fn get(k: K) -> V {
            ret option::get(get(*tbl, k));
        }

        fn find(k: K) -> option::t<V> {
            ret get(*tbl, k);
        }

        fn remove(k: K) -> option::t<V> {
            ret remove(*tbl, k);
        }

        fn rehash() {
            rehash(*tbl);
        }

        fn items(blk: block(K, V)) {
            items(*tbl, blk);
        }

        fn keys(blk: block(K)) {
            items(*tbl) { |k, _v| blk(k) }
        }

        fn values(blk: block(V)) {
            items(*tbl) { |_k, v| blk(v) }
        }
    }

    fn mk<copy K, copy V>(hasher: hashfn<K>, eqer: eqfn<K>) -> hashmap<K,V> {
        let initial_capacity: uint = 32u; // 2^5
        let t = @{mutable size: 0u,
                  mutable chains: chains(initial_capacity),
                  hasher: hasher,
                  eqer: eqer};
        ret o(t, 0.75);
    }
}

/*
Function: mk_hashmap

Construct a hashmap

Parameters:

hasher - The hash function for key type K
eqer - The equality function for key type K
*/
fn mk_flat_hashmap<copy K, copy V>(hasher: hashfn<K>, eqer: eqfn<K>)
    -> hashmap<K, V> {
    let initial_capacity: uint = 32u; // 2^5

    let load_factor: util::rational = {num: 3, den: 4};
    tag bucket<copy K, copy V> { nil; deleted; some(K, V); }
    fn make_buckets<copy K, copy V>(nbkts: uint) -> [mutable bucket<K, V>] {
        ret vec::init_elt_mut::<bucket<K, V>>(nil::<K, V>, nbkts);
    }
    // Derive two hash functions from the one given by taking the upper
    // half and lower half of the uint bits.  Our bucket probing
    // sequence is then defined by
    //
    //   hash(key, i) := hashl(key) * i + hashr(key)   for i = 0, 1, 2, ...
    //
    // Tearing the hash function apart this way is kosher in practice
    // as, assuming 32-bit uints, the table would have to be at 2^32
    // buckets before the resulting pair of hash functions no longer
    // probes all buckets for a fixed key.  Note that hashl is made to
    // output odd numbers (hence coprime to the number of nbkts, which
    // is always a power? of 2), so that all buckets are probed for a
    // fixed key.

    fn hashl(n: u32) -> u32 { ret (n >>> 16u32) * 2u32 + 1u32; }
    fn hashr(n: u32) -> u32 { ret 0x0000_ffff_u32 & n; }
    fn hash(h: u32, nbkts: uint, i: uint) -> uint {
        ret ((hashl(h) as uint) * i + (hashr(h) as uint)) % nbkts;
    }

    fn to_u64(h: uint) -> u32 {
        ret (h as u32) ^ ((h >>> 16u) as u32);
    }

    /**
     * We attempt to never call this with a full table.  If we do, it
     * will fail.
     */
    fn insert_common<copy K, copy V>(hasher: hashfn<K>, eqer: eqfn<K>,
                                     bkts: [mutable bucket<K, V>],
                                     nbkts: uint, key: K, val: V) -> bool {
        let i: uint = 0u;
        let h = to_u64(hasher(key));
        while i < nbkts {
            let j: uint = hash(h, nbkts, i);
            alt bkts[j] {
              some(k, _) {
                // Copy key to please alias analysis.

                let k_ = k;
                if eqer(key, k_) { bkts[j] = some(k_, val); ret false; }
                i += 1u;
              }
              _ { bkts[j] = some(key, val); ret true; }
            }
        }
        fail; // full table
    }
    fn find_common<copy K, copy V>(hasher: hashfn<K>, eqer: eqfn<K>,
                                   bkts: [mutable bucket<K, V>],
                                   nbkts: uint, key: K) -> option::t<V> {
        let i: uint = 0u;
        let h = to_u64(hasher(key));
        while i < nbkts {
            let j: uint = hash(h, nbkts, i);
            alt bkts[j] {
              some(k, v) {
                // Copy to please alias analysis.
                let k_ = k;
                let v_ = v;
                if eqer(key, k_) { ret option::some(v_); }
              }
              nil. { ret option::none; }
              deleted. { }
            }
            i += 1u;
        }
        ret option::none;
    }
    fn rehash<copy K, copy V>(hasher: hashfn<K>, eqer: eqfn<K>,
                              oldbkts: [mutable bucket<K, V>],
                              _noldbkts: uint,
                              newbkts: [mutable bucket<K, V>],
                              nnewbkts: uint) {
        for b: bucket<K, V> in oldbkts {
            alt b {
              some(k_, v_) {
                let k = k_;
                let v = v_;
                insert_common(hasher, eqer, newbkts, nnewbkts, k, v);
              }
              _ { }
            }
        }
    }
    obj hashmap<copy K, copy V>(hasher: hashfn<K>,
                                eqer: eqfn<K>,
                                mutable bkts: [mutable bucket<K, V>],
                                mutable nbkts: uint,
                                mutable nelts: uint,
                                lf: util::rational) {
        fn size() -> uint { ret nelts; }
        fn insert(key: K, val: V) -> bool {
            let load: util::rational =
                {num: nelts + 1u as int, den: nbkts as int};
            if !util::rational_leq(load, lf) {
                let nnewbkts: uint = uint::next_power_of_two(nbkts + 1u);
                let newbkts = make_buckets(nnewbkts);
                rehash(hasher, eqer, bkts, nbkts, newbkts, nnewbkts);
                bkts = newbkts;
                nbkts = nnewbkts;
            }
            if insert_common(hasher, eqer, bkts, nbkts, key, val) {
                nelts += 1u;
                ret true;
            }
            ret false;
        }
        fn contains_key(key: K) -> bool {
            ret alt find_common(hasher, eqer, bkts, nbkts, key) {
                  option::some(_) { true }
                  _ { false }
                };
        }
        fn get(key: K) -> V {
            ret alt find_common(hasher, eqer, bkts, nbkts, key) {
                  option::some(val) { val }
                  _ { fail }
                };
        }
        fn find(key: K) -> option::t<V> {
            be find_common(hasher, eqer, bkts, nbkts, key);
        }
        fn remove(key: K) -> option::t<V> {
            let i: uint = 0u;
            let h = to_u64(hasher(key));
            while i < nbkts {
                let j: uint = hash(h, nbkts, i);
                alt bkts[j] {
                  some(k, v) {
                    let k_ = k;
                    let vo = option::some(v);
                    if eqer(key, k_) {
                        bkts[j] = deleted;
                        nelts -= 1u;
                        ret vo;
                    }
                  }
                  deleted. { }
                  nil. { ret option::none; }
                }
                i += 1u;
            }
            ret option::none;
        }
        fn rehash() {
            let newbkts = make_buckets(nbkts);
            rehash(hasher, eqer, bkts, nbkts, newbkts, nbkts);
            bkts = newbkts;
        }
        fn items(it: block(K, V)) {
            for b in bkts {
                alt b { some(k, v) { it(copy k, copy v); } _ { } }
            }
        }
        fn keys(it: block(K)) {
            for b in bkts {
                alt b { some(k, _) { it(copy k); } _ { } }
            }
        }
        fn values(it: block(V)) {
            for b in bkts {
                alt b { some(_, v) { it(copy v); } _ { } }
            }
        }
    }
    let bkts = make_buckets(initial_capacity);
    ret hashmap(hasher, eqer, bkts, initial_capacity, 0u, load_factor);
}

fn mk_hashmap<copy K, copy V>(hasher: hashfn<K>, eqer: eqfn<K>)
    -> hashmap<K, V> {
    ret chained::mk(hasher, eqer);
}

/*
Function: new_str_hash

Construct a hashmap for string keys
*/
fn new_str_hash<copy V>() -> hashmap<str, V> {
    ret mk_hashmap(str::hash, str::eq);
}

/*
Function: new_int_hash

Construct a hashmap for int keys
*/
fn new_int_hash<copy V>() -> hashmap<int, V> {
    fn hash_int(&&x: int) -> uint { ret x as uint; }
    fn eq_int(&&a: int, &&b: int) -> bool { ret a == b; }
    ret mk_hashmap(hash_int, eq_int);
}

/*
Function: new_uint_hash

Construct a hashmap for uint keys
*/
fn new_uint_hash<copy V>() -> hashmap<uint, V> {
    fn hash_uint(&&x: uint) -> uint { ret x; }
    fn eq_uint(&&a: uint, &&b: uint) -> bool { ret a == b; }
    ret mk_hashmap(hash_uint, eq_uint);
}

/*
Function: set_add

Convenience function for adding keys to a hashmap with nil type keys
*/
fn set_add<K>(set: hashset<K>, key: K) -> bool { ret set.insert(key, ()); }

// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
