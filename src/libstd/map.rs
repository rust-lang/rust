/*
Module: map

A hashmap
*/

/* Section: Types */

/*
Type: hashfn

A function that returns a hash of a value.
The hash should concentrate entropy in the
lower bits.
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
    fn find(K) -> core::option::t<V>;
    /*
    Method: remove

    Remove and return a value from the map. If the key does not exist
    in the map then returns none.
    */
    fn remove(K) -> core::option::t<V>;
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
    type entry<K: copy, V: copy> = {
        hash: uint,
        key: K,
        mutable value: V,
        mutable next: chain<K, V>
    };

    tag chain<K: copy, V: copy> {
        present(@entry<K, V>);
        absent;
    }

    type t<K: copy, V: copy> = {
        mutable size: uint,
        mutable chains: [mutable chain<K,V>],
        hasher: hashfn<K>,
        eqer: eqfn<K>
    };

    tag search_result<K: copy, V: copy> {
        not_found;
        found_first(uint, @entry<K,V>);
        found_after(@entry<K,V>, @entry<K,V>);
    }

    fn search_rem<K: copy, V: copy>(tbl: t<K,V>,
                                  k: K,
                                  h: uint,
                                  idx: uint,
                                  e_root: @entry<K,V>) -> search_result<K,V> {
        let e0 = e_root;
        let comp = 1u;   // for logging
        while true {
            alt e0.next {
              absent. {
                #debug("search_tbl: absent, comp %u, hash %u, idx %u",
                       comp, h, idx);
                ret not_found;
              }
              present(e1) {
                comp += 1u;
                let e1_key = e1.key; // Satisfy alias checker.
                if e1.hash == h && tbl.eqer(e1_key, k) {
                    #debug("search_tbl: present, comp %u, hash %u, idx %u",
                           comp, h, idx);
                    ret found_after(e0, e1);
                } else {
                    e0 = e1;
                }
              }
            }
        }
        util::unreachable();
    }

    fn search_tbl<K: copy, V: copy>(
        tbl: t<K,V>, k: K, h: uint) -> search_result<K,V> {
        let idx = h % vec::len(tbl.chains);
        alt tbl.chains[idx] {
          absent. {
            #debug("search_tbl: absent, comp %u, hash %u, idx %u",
                   0u, h, idx);
            ret not_found;
          }
          present(e) {
            let e_key = e.key; // Satisfy alias checker.
            if e.hash == h && tbl.eqer(e_key, k) {
                #debug("search_tbl: present, comp %u, hash %u, idx %u",
                       1u, h, idx);
                ret found_first(idx, e);
            } else {
                ret search_rem(tbl, k, h, idx, e);
            }
          }
        }
    }

    fn insert<K: copy, V: copy>(tbl: t<K,V>, k: K, v: V) -> bool {
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

    fn get<K: copy, V: copy>(tbl: t<K,V>, k: K) -> core::option::t<V> {
        alt search_tbl(tbl, k, tbl.hasher(k)) {
          not_found. {
            ret core::option::none;
          }

          found_first(_, entry) {
            ret core::option::some(entry.value);
          }

          found_after(_, entry) {
            ret core::option::some(entry.value);
          }
        }
    }

    fn remove<K: copy, V: copy>(tbl: t<K,V>, k: K) -> core::option::t<V> {
        alt search_tbl(tbl, k, tbl.hasher(k)) {
          not_found. {
            ret core::option::none;
          }

          found_first(idx, entry) {
            tbl.size -= 1u;
            tbl.chains[idx] = entry.next;
            ret core::option::some(entry.value);
          }

          found_after(eprev, entry) {
            tbl.size -= 1u;
            eprev.next = entry.next;
            ret core::option::some(entry.value);
          }
        }
    }

    fn chains<K: copy, V: copy>(nchains: uint) -> [mutable chain<K,V>] {
        ret vec::init_elt_mut(absent, nchains);
    }

    fn foreach_entry<K: copy, V: copy>(chain0: chain<K,V>,
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

    fn foreach_chain<K: copy, V: copy>(chains: [const chain<K,V>],
                                     blk: block(@entry<K,V>)) {
        let i = 0u, n = vec::len(chains);
        while i < n {
            foreach_entry(chains[i], blk);
            i += 1u;
        }
    }

    fn rehash<K: copy, V: copy>(tbl: t<K,V>) {
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

    fn items<K: copy, V: copy>(tbl: t<K,V>, blk: block(K,V)) {
        let tbl_chains = tbl.chains;  // Satisfy alias checker.
        foreach_chain(tbl_chains) { |entry|
            let key = entry.key;
            let value = entry.value;
            blk(key, value);
        }
    }

    obj o<K: copy, V: copy>(tbl: @t<K,V>,
                          lf: util::rational) {
        fn size() -> uint {
            ret tbl.size;
        }

        fn insert(k: K, v: V) -> bool {
            let nchains = vec::len(tbl.chains);
            let load = {num:tbl.size + 1u as int, den:nchains as int};
            if !util::rational_leq(load, lf) {
                rehash(*tbl);
            }
            ret insert(*tbl, k, v);
        }

        fn contains_key(k: K) -> bool {
            ret core::option::is_some(get(*tbl, k));
        }

        fn get(k: K) -> V {
            ret core::option::get(get(*tbl, k));
        }

        fn find(k: K) -> core::option::t<V> {
            ret get(*tbl, k);
        }

        fn remove(k: K) -> core::option::t<V> {
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

    fn mk<K: copy, V: copy>(hasher: hashfn<K>, eqer: eqfn<K>)
        -> hashmap<K,V> {
        let initial_capacity: uint = 32u; // 2^5
        let t = @{mutable size: 0u,
                  mutable chains: chains(initial_capacity),
                  hasher: hasher,
                  eqer: eqer};
        ret o(t, {num:3, den:4});
    }
}

/*
Function: mk_hashmap

Construct a hashmap.

Parameters:

hasher - The hash function for key type K
eqer - The equality function for key type K
*/
fn mk_hashmap<K: copy, V: copy>(hasher: hashfn<K>, eqer: eqfn<K>)
    -> hashmap<K, V> {
    ret chained::mk(hasher, eqer);
}

/*
Function: new_str_hash

Construct a hashmap for string keys
*/
fn new_str_hash<V: copy>() -> hashmap<str, V> {
    ret mk_hashmap(str::hash, str::eq);
}

/*
Function: new_int_hash

Construct a hashmap for int keys
*/
fn new_int_hash<V: copy>() -> hashmap<int, V> {
    fn hash_int(&&x: int) -> uint { ret x as uint; }
    fn eq_int(&&a: int, &&b: int) -> bool { ret a == b; }
    ret mk_hashmap(hash_int, eq_int);
}

/*
Function: new_uint_hash

Construct a hashmap for uint keys
*/
fn new_uint_hash<V: copy>() -> hashmap<uint, V> {
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
