/*
Module: map

A map type
*/

/* Section: Types */

/*
Type: hashfn

A function that returns a hash of a value.
The hash should concentrate entropy in the
lower bits.
*/
type hashfn<K> = fn@(K) -> uint;

/*
Type: eqfn

Equality
*/
type eqfn<K> = fn@(K, K) -> bool;

/*
Type: hashset

A convenience type to treat a map as a set
*/
type set<K> = map<K, ()>;

// Temporary alias to make migration easier
type hashmap<K, V> = map<K, V>;

/*
IFace: map
*/
iface map<K: copy, V: copy> {
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
}

// FIXME: package this up and export it as a datatype usable for
// external code that doesn't want to pay the cost of a box and vtable
// lookups.
mod chained {
    type entry<K, V> = {
        hash: uint,
        key: K,
        mutable value: V,
        mutable next: chain<K, V>
    };

    tag chain<K, V> {
        present(@entry<K, V>);
        absent;
    }

    type t<K, V> = {
        mutable size: uint,
        mutable chains: [mutable chain<K,V>],
        hasher: hashfn<K>,
        eqer: eqfn<K>
    };

    tag search_result<K, V> {
        not_found;
        found_first(uint, @entry<K,V>);
        found_after(@entry<K,V>, @entry<K,V>);
    }

    fn search_rem<K: copy, V: copy>(
        tbl: t<K,V>, k: K, h: uint, idx: uint,
        e_root: @entry<K,V>) -> search_result<K,V> {
        let e0 = e_root;
        let comp = 1u;   // for logging
        while true {
            alt e0.next {
              absent {
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
          absent {
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
          not_found {
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
          not_found {
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
          not_found {
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
              absent { ret; }
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

    impl <K: copy, V: copy> of map<K, V> for t<K, V> {
        fn size() -> uint { self.size }

        fn insert(k: K, v: V) -> bool {
            let nchains = vec::len(self.chains);
            let load = {num: self.size + 1u as int, den: nchains as int};
            // Structural consts would be nice. This is a const 3/4
            // load factor that we compare against.
            if !util::rational_leq(load, {num:3, den:4}) { rehash(self); }
            ret insert(self, k, v);
        }

        fn contains_key(k: K) -> bool { option::is_some(get(self, k)) }

        fn get(k: K) -> V { option::get(get(self, k)) }

        fn find(k: K) -> option::t<V> { get(self, k) }

        fn remove(k: K) -> option::t<V> { remove(self, k) }

        fn items(blk: block(K, V)) { items(self, blk); }

        fn keys(blk: block(K)) { items(self) { |k, _v| blk(k) } }

        fn values(blk: block(V)) { items(self) { |_k, v| blk(v) } }
    }

    fn mk<K: copy, V: copy>(hasher: hashfn<K>, eqer: eqfn<K>) -> map<K,V> {
        let initial_capacity: uint = 32u; // 2^5
        let slf: t<K, V> = {mutable size: 0u,
                            mutable chains: chains(initial_capacity),
                            hasher: hasher,
                            eqer: eqer};
        slf as map::<K, V>
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
    -> map<K, V> {
    chained::mk(hasher, eqer)
}

/*
Function: new_str_hash

Construct a hashmap for string keys
*/
fn new_str_hash<V: copy>() -> map<str, V> {
    ret mk_hashmap(str::hash, str::eq);
}

/*
Function: new_bytes_hash

Construct a hashmap for byte string keys
*/
fn new_bytes_hash<V: copy>() -> map<[u8], V> {
    ret mk_hashmap(vec::u8::hash, vec::u8::eq);
}

/*
Function: new_int_hash

Construct a hashmap for int keys
*/
fn new_int_hash<V: copy>() -> map<int, V> {
    fn hash_int(&&x: int) -> uint { int::hash(x) }
    fn eq_int(&&a: int, &&b: int) -> bool { ret a == b; }
    ret mk_hashmap(hash_int, eq_int);
}

/*
Function: new_uint_hash

Construct a hashmap for uint keys
*/
fn new_uint_hash<V: copy>() -> map<uint, V> {
    fn hash_uint(&&x: uint) -> uint { uint::hash(x) }
    fn eq_uint(&&a: uint, &&b: uint) -> bool { ret a == b; }
    ret mk_hashmap(hash_uint, eq_uint);
}

/*
Function: set_add

Convenience function for adding keys to a hashmap with nil type keys
*/
fn set_add<K>(set: set<K>, key: K) -> bool { ret set.insert(key, ()); }

#[cfg(test)]
mod tests {

    #[test]
    fn test_simple() {
        #debug("*** starting test_simple");
        fn eq_uint(&&x: uint, &&y: uint) -> bool { ret x == y; }
        fn uint_id(&&x: uint) -> uint { x }
        let hasher_uint: map::hashfn<uint> = uint_id;
        let eqer_uint: map::eqfn<uint> = eq_uint;
        let hasher_str: map::hashfn<str> = str::hash;
        let eqer_str: map::eqfn<str> = str::eq;
        #debug("uint -> uint");
        let hm_uu: map::hashmap<uint, uint> =
            map::mk_hashmap::<uint, uint>(hasher_uint, eqer_uint);
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
        let ten: str = "ten";
        let eleven: str = "eleven";
        let twelve: str = "twelve";
        #debug("str -> uint");
        let hm_su: map::hashmap<str, uint> =
            map::mk_hashmap::<str, uint>(hasher_str, eqer_str);
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
        #debug("uint -> str");
        let hm_us: map::hashmap<uint, str> =
            map::mk_hashmap::<uint, str>(hasher_uint, eqer_uint);
        assert (hm_us.insert(10u, "twelve"));
        assert (hm_us.insert(11u, "thirteen"));
        assert (hm_us.insert(12u, "fourteen"));
        assert (str::eq(hm_us.get(11u), "thirteen"));
        assert (str::eq(hm_us.get(12u), "fourteen"));
        assert (str::eq(hm_us.get(10u), "twelve"));
        assert (!hm_us.insert(12u, "fourteen"));
        assert (str::eq(hm_us.get(12u), "fourteen"));
        assert (!hm_us.insert(12u, "twelve"));
        assert (str::eq(hm_us.get(12u), "twelve"));
        #debug("str -> str");
        let hm_ss: map::hashmap<str, str> =
            map::mk_hashmap::<str, str>(hasher_str, eqer_str);
        assert (hm_ss.insert(ten, "twelve"));
        assert (hm_ss.insert(eleven, "thirteen"));
        assert (hm_ss.insert(twelve, "fourteen"));
        assert (str::eq(hm_ss.get("eleven"), "thirteen"));
        assert (str::eq(hm_ss.get("twelve"), "fourteen"));
        assert (str::eq(hm_ss.get("ten"), "twelve"));
        assert (!hm_ss.insert("twelve", "fourteen"));
        assert (str::eq(hm_ss.get("twelve"), "fourteen"));
        assert (!hm_ss.insert("twelve", "twelve"));
        assert (str::eq(hm_ss.get("twelve"), "twelve"));
        #debug("*** finished test_simple");
    }


    /**
    * Force map growth
    */
    #[test]
    fn test_growth() {
        #debug("*** starting test_growth");
        let num_to_insert: uint = 64u;
        fn eq_uint(&&x: uint, &&y: uint) -> bool { ret x == y; }
        fn uint_id(&&x: uint) -> uint { x }
        #debug("uint -> uint");
        let hasher_uint: map::hashfn<uint> = uint_id;
        let eqer_uint: map::eqfn<uint> = eq_uint;
        let hm_uu: map::hashmap<uint, uint> =
            map::mk_hashmap::<uint, uint>(hasher_uint, eqer_uint);
        let i: uint = 0u;
        while i < num_to_insert {
            assert (hm_uu.insert(i, i * i));
            #debug("inserting %u -> %u", i, i*i);
            i += 1u;
        }
        #debug("-----");
        i = 0u;
        while i < num_to_insert {
            #debug("get(%u) = %u", i, hm_uu.get(i));
            assert (hm_uu.get(i) == i * i);
            i += 1u;
        }
        assert (hm_uu.insert(num_to_insert, 17u));
        assert (hm_uu.get(num_to_insert) == 17u);
        #debug("-----");
        i = 0u;
        while i < num_to_insert {
            #debug("get(%u) = %u", i, hm_uu.get(i));
            assert (hm_uu.get(i) == i * i);
            i += 1u;
        }
        #debug("str -> str");
        let hasher_str: map::hashfn<str> = str::hash;
        let eqer_str: map::eqfn<str> = str::eq;
        let hm_ss: map::hashmap<str, str> =
            map::mk_hashmap::<str, str>(hasher_str, eqer_str);
        i = 0u;
        while i < num_to_insert {
            assert hm_ss.insert(uint::to_str(i, 2u), uint::to_str(i * i, 2u));
            #debug("inserting \"%s\" -> \"%s\"",
                   uint::to_str(i, 2u),
                   uint::to_str(i*i, 2u));
            i += 1u;
        }
        #debug("-----");
        i = 0u;
        while i < num_to_insert {
            #debug("get(\"%s\") = \"%s\"",
                   uint::to_str(i, 2u),
                   hm_ss.get(uint::to_str(i, 2u)));
            assert (str::eq(hm_ss.get(uint::to_str(i, 2u)),
                            uint::to_str(i * i, 2u)));
            i += 1u;
        }
        assert (hm_ss.insert(uint::to_str(num_to_insert, 2u),
                             uint::to_str(17u, 2u)));
        assert (str::eq(hm_ss.get(uint::to_str(num_to_insert, 2u)),
                        uint::to_str(17u, 2u)));
        #debug("-----");
        i = 0u;
        while i < num_to_insert {
            #debug("get(\"%s\") = \"%s\"",
                   uint::to_str(i, 2u),
                   hm_ss.get(uint::to_str(i, 2u)));
            assert (str::eq(hm_ss.get(uint::to_str(i, 2u)),
                            uint::to_str(i * i, 2u)));
            i += 1u;
        }
        #debug("*** finished test_growth");
    }

    #[test]
    fn test_removal() {
        #debug("*** starting test_removal");
        let num_to_insert: uint = 64u;
        fn eq(&&x: uint, &&y: uint) -> bool { ret x == y; }
        fn hash(&&u: uint) -> uint {
            // This hash function intentionally causes collisions between
            // consecutive integer pairs.

            ret u / 2u * 2u;
        }
        assert (hash(0u) == hash(1u));
        assert (hash(2u) == hash(3u));
        assert (hash(0u) != hash(2u));
        let hasher: map::hashfn<uint> = hash;
        let eqer: map::eqfn<uint> = eq;
        let hm: map::hashmap<uint, uint> =
            map::mk_hashmap::<uint, uint>(hasher, eqer);
        let i: uint = 0u;
        while i < num_to_insert {
            assert (hm.insert(i, i * i));
            #debug("inserting %u -> %u", i, i*i);
            i += 1u;
        }
        assert (hm.size() == num_to_insert);
        #debug("-----");
        #debug("removing evens");
        i = 0u;
        while i < num_to_insert {
            let v = hm.remove(i);
            alt v {
              option::some(u) { assert (u == i * i); }
              option::none { fail; }
            }
            i += 2u;
        }
        assert (hm.size() == num_to_insert / 2u);
        #debug("-----");
        i = 1u;
        while i < num_to_insert {
            #debug("get(%u) = %u", i, hm.get(i));
            assert (hm.get(i) == i * i);
            i += 2u;
        }
        #debug("-----");
        i = 1u;
        while i < num_to_insert {
            #debug("get(%u) = %u", i, hm.get(i));
            assert (hm.get(i) == i * i);
            i += 2u;
        }
        #debug("-----");
        i = 0u;
        while i < num_to_insert {
            assert (hm.insert(i, i * i));
            #debug("inserting %u -> %u", i, i*i);
            i += 2u;
        }
        assert (hm.size() == num_to_insert);
        #debug("-----");
        i = 0u;
        while i < num_to_insert {
            #debug("get(%u) = %u", i, hm.get(i));
            assert (hm.get(i) == i * i);
            i += 1u;
        }
        #debug("-----");
        assert (hm.size() == num_to_insert);
        i = 0u;
        while i < num_to_insert {
            #debug("get(%u) = %u", i, hm.get(i));
            assert (hm.get(i) == i * i);
            i += 1u;
        }
        #debug("*** finished test_removal");
    }

    #[test]
    fn test_contains_key() {
        let key = "k";
        let map = map::mk_hashmap::<str, str>(str::hash, str::eq);
        assert (!map.contains_key(key));
        map.insert(key, "val");
        assert (map.contains_key(key));
    }

    #[test]
    fn test_find() {
        let key = "k";
        let map = map::mk_hashmap::<str, str>(str::hash, str::eq);
        assert (option::is_none(map.find(key)));
        map.insert(key, "val");
        assert (option::get(map.find(key)) == "val");
    }
}