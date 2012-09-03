//! A map type

#[forbid(deprecated_mode)];
#[forbid(deprecated_pattern)];

import io::WriterUtil;
import to_str::ToStr;
import managed::Managed;
import send_map::linear::LinearMap;

import core::cmp::Eq;
import hash::Hash;
import to_bytes::IterBytes;

export hashmap, hashfn, eqfn, set, map, chained, hashmap, str_hash;
export box_str_hash;
export bytes_hash, int_hash, uint_hash, set_add;
export hash_from_vec, hash_from_strs, hash_from_bytes;
export hash_from_ints, hash_from_uints;
export vec_from_set;

/**
 * A function that returns a hash of a value
 *
 * The hash should concentrate entropy in the lower bits.
 */
type hashfn<K> = fn~(key: &K) -> uint;

type eqfn<K> = fn~(key1: &K, key2: &K) -> bool;

/// A convenience type to treat a hashmap as a set
type set<K> = hashmap<K, ()>;

type hashmap<K, V> = chained::t<K, V>;

trait map<K: copy, V: copy> {
    /// Return the number of elements in the map
    pure fn size() -> uint;

    /**
     * Add a value to the map.
     *
     * If the map already contains a value for the specified key then the
     * original value is replaced.
     *
     * Returns true if the key did not already exist in the map
     */
    fn insert(+K, +V) -> bool;

    /// Returns true if the map contains a value for the specified key
    fn contains_key(+key: K) -> bool;

    /// Returns true if the map contains a value for the specified
    /// key, taking the key by reference.
    fn contains_key_ref(key: &K) -> bool;

    /**
     * Get the value for the specified key. Fails if the key does not exist in
     * the map.
     */
    fn get(+key: K) -> V;

    /**
     * Get the value for the specified key. If the key does not exist in
     * the map then returns none.
     */
    pure fn find(+key: K) -> Option<V>;

    /**
     * Remove and return a value from the map. Returns true if the
     * key was present in the map, otherwise false.
     */
    fn remove(+key: K) -> bool;

    /// Clear the map, removing all key/value pairs.
    fn clear();

    /// Iterate over all the key/value pairs in the map by value
    pure fn each(fn(+key: K, +value: V) -> bool);

    /// Iterate over all the keys in the map by value
    pure fn each_key(fn(+key: K) -> bool);

    /// Iterate over all the values in the map by value
    pure fn each_value(fn(+value: V) -> bool);

    /// Iterate over all the key/value pairs in the map by reference
    pure fn each_ref(fn(key: &K, value: &V) -> bool);

    /// Iterate over all the keys in the map by reference
    pure fn each_key_ref(fn(key: &K) -> bool);

    /// Iterate over all the values in the map by reference
    pure fn each_value_ref(fn(value: &V) -> bool);
}

mod util {
    type rational = {num: int, den: int}; // : int::positive(*.den);

    pure fn rational_leq(x: rational, y: rational) -> bool {
        // NB: Uses the fact that rationals have positive denominators WLOG:

        x.num * y.den <= y.num * x.den
    }
}


// FIXME (#2344): package this up and export it as a datatype usable for
// external code that doesn't want to pay the cost of a box.
mod chained {
    export t, mk, hashmap;

    const initial_capacity: uint = 32u; // 2^5

    struct entry<K, V> {
        hash: uint;
        key: K;
        value: V;
        mut next: Option<@entry<K, V>>;
    }

    struct hashmap_<K, V> {
        mut count: uint;
        mut chains: ~[mut Option<@entry<K,V>>];
        hasher: hashfn<K>;
        eqer: eqfn<K>;
    }

    type t<K, V> = @hashmap_<K, V>;

    enum search_result<K, V> {
        not_found,
        found_first(uint, @entry<K,V>),
        found_after(@entry<K,V>, @entry<K,V>)
    }

    priv impl<K, V: copy> t<K, V> {
        pure fn search_rem(k: &K, h: uint, idx: uint,
                           e_root: @entry<K,V>) -> search_result<K,V> {
            let mut e0 = e_root;
            let mut comp = 1u;   // for logging
            loop {
                match copy e0.next {
                  None => {
                    debug!("search_tbl: absent, comp %u, hash %u, idx %u",
                           comp, h, idx);
                    return not_found;
                  }
                  Some(e1) => {
                    comp += 1u;
                    unchecked {
                        if e1.hash == h && self.eqer(&e1.key, k) {
                            debug!("search_tbl: present, comp %u, \
                                    hash %u, idx %u",
                                   comp, h, idx);
                            return found_after(e0, e1);
                        } else {
                            e0 = e1;
                        }
                    }
                  }
                }
            };
        }

        pure fn search_tbl(k: &K, h: uint) -> search_result<K,V> {
            let idx = h % vec::len(self.chains);
            match copy self.chains[idx] {
              None => {
                debug!("search_tbl: none, comp %u, hash %u, idx %u",
                       0u, h, idx);
                return not_found;
              }
              Some(e) => {
                unchecked {
                    if e.hash == h && self.eqer(&e.key, k) {
                        debug!("search_tbl: present, comp %u, hash %u, \
                                idx %u", 1u, h, idx);
                        return found_first(idx, e);
                    } else {
                        return self.search_rem(k, h, idx, e);
                    }
                }
              }
            }
        }

        fn rehash() {
            let n_old_chains = self.chains.len();
            let n_new_chains: uint = uint::next_power_of_two(n_old_chains+1u);
            let new_chains = chains(n_new_chains);
            for self.each_entry |entry| {
                let idx = entry.hash % n_new_chains;
                entry.next = new_chains[idx];
                new_chains[idx] = Some(entry);
            }
            self.chains = new_chains;
        }

        pure fn each_entry(blk: fn(@entry<K,V>) -> bool) {
            // n.b. we can't use vec::iter() here because self.chains
            // is stored in a mutable location.
            let mut i = 0u, n = self.chains.len();
            while i < n {
                let mut chain = self.chains[i];
                loop {
                    chain = match chain {
                      None => break,
                      Some(entry) => {
                        let next = entry.next;
                        if !blk(entry) { return; }
                        next
                      }
                    }
                }
                i += 1u;
            }
        }
    }

    impl<K: copy, V: copy> t<K, V>: map<K, V> {
        pure fn size() -> uint { self.count }

        fn contains_key(+k: K) -> bool {
            self.contains_key_ref(&k)
        }

        fn contains_key_ref(k: &K) -> bool {
            let hash = self.hasher(k);
            match self.search_tbl(k, hash) {
              not_found => false,
              found_first(*) | found_after(*) => true
            }
        }

        fn insert(+k: K, +v: V) -> bool {
            let hash = self.hasher(&k);
            match self.search_tbl(&k, hash) {
              not_found => {
                self.count += 1u;
                let idx = hash % vec::len(self.chains);
                let old_chain = self.chains[idx];
                self.chains[idx] = Some(@entry {
                    hash: hash,
                    key: k,
                    value: v,
                    next: old_chain});

                // consider rehashing if more 3/4 full
                let nchains = vec::len(self.chains);
                let load = {num: (self.count + 1u) as int,
                            den: nchains as int};
                if !util::rational_leq(load, {num:3, den:4}) {
                    self.rehash();
                }

                return true;
              }
              found_first(idx, entry) => {
                self.chains[idx] = Some(@entry {
                    hash: hash,
                    key: k,
                    value: v,
                    next: entry.next});
                return false;
              }
              found_after(prev, entry) => {
                prev.next = Some(@entry {
                    hash: hash,
                    key: k,
                    value: v,
                    next: entry.next});
                return false;
              }
            }
        }

        pure fn find(+k: K) -> Option<V> {
            unchecked {
                match self.search_tbl(&k, self.hasher(&k)) {
                  not_found => None,
                  found_first(_, entry) => Some(entry.value),
                  found_after(_, entry) => Some(entry.value)
                }
            }
        }

        fn get(+k: K) -> V {
            let opt_v = self.find(k);
            if opt_v.is_none() {
                fail fmt!("Key not found in table: %?", k);
            }
            option::unwrap(opt_v)
        }

        fn remove(+k: K) -> bool {
            match self.search_tbl(&k, self.hasher(&k)) {
              not_found => false,
              found_first(idx, entry) => {
                self.count -= 1u;
                self.chains[idx] = entry.next;
                true
              }
              found_after(eprev, entry) => {
                self.count -= 1u;
                eprev.next = entry.next;
                true
              }
            }
        }

        fn clear() {
            self.count = 0u;
            self.chains = chains(initial_capacity);
        }

        pure fn each(blk: fn(+key: K, +value: V) -> bool) {
            self.each_ref(|k, v| blk(*k, *v))
        }

        pure fn each_key(blk: fn(+key: K) -> bool) {
            self.each_key_ref(|p| blk(*p))
        }

        pure fn each_value(blk: fn(+value: V) -> bool) {
            self.each_value_ref(|p| blk(*p))
        }

        pure fn each_ref(blk: fn(key: &K, value: &V) -> bool) {
            for self.each_entry |entry| {
                if !blk(&entry.key, &entry.value) { break; }
            }
        }

        pure fn each_key_ref(blk: fn(key: &K) -> bool) {
            self.each_ref(|k, _v| blk(k))
        }

        pure fn each_value_ref(blk: fn(value: &V) -> bool) {
            self.each_ref(|_k, v| blk(v))
        }
    }

    impl<K: copy ToStr, V: ToStr copy> t<K, V>: ToStr {
        fn to_writer(wr: io::Writer) {
            if self.count == 0u {
                wr.write_str(~"{}");
                return;
            }

            wr.write_str(~"{ ");
            let mut first = true;
            for self.each_entry |entry| {
                if !first {
                    wr.write_str(~", ");
                }
                first = false;
                wr.write_str(entry.key.to_str());
                wr.write_str(~": ");
                wr.write_str((copy entry.value).to_str());
            };
            wr.write_str(~" }");
        }

        fn to_str() -> ~str {
            do io::with_str_writer |wr| { self.to_writer(wr) }
        }
    }

    impl<K: copy, V: copy> t<K, V>: ops::Index<K, V> {
        pure fn index(&&k: K) -> V {
            unchecked {
                self.get(k)
            }
        }
    }

    fn chains<K,V>(nchains: uint) -> ~[mut Option<@entry<K,V>>] {
        vec::to_mut(vec::from_elem(nchains, None))
    }

    fn mk<K, V: copy>(+hasher: hashfn<K>, +eqer: eqfn<K>) -> t<K,V> {
        let slf: t<K, V> = @hashmap_ {count: 0u,
                                      chains: chains(initial_capacity),
                                      hasher: hasher,
                                      eqer: eqer};
        slf
    }
}

/*
Function: hashmap

Construct a hashmap.

Parameters:

hasher - The hash function for key type K
eqer - The equality function for key type K
*/
fn hashmap<K: const, V: copy>(+hasher: hashfn<K>, +eqer: eqfn<K>)
        -> hashmap<K, V> {
    chained::mk(hasher, eqer)
}

/// Construct a hashmap for string-slice keys
fn str_slice_hash<V: copy>() -> hashmap<&str, V> {
    return hashmap(|s| hash::hash_str(*s) as uint,
                   |a,b| str::eq_slice(*a, *b));
}

/// Construct a hashmap for string keys
fn str_hash<V: copy>() -> hashmap<~str, V> {
    return hashmap(str::hash, str::eq);
}

/// Construct a hashmap for boxed string keys
fn box_str_hash<V: copy>() -> hashmap<@~str, V> {
    hashmap(|x: &@~str| str::hash(&**x),
            |x: &@~str, y: &@~str| str::eq(&**x, &**y))
}

/// Construct a hashmap for byte string keys
fn bytes_hash<V: copy>() -> hashmap<~[u8], V> {
    return hashmap(vec::u8::hash, vec::u8::eq);
}

/// Construct a hashmap for int keys
fn int_hash<V: copy>() -> hashmap<int, V> {
   return hashmap(|x| { int::hash(*x) }, |x, y| { int::eq(*x, *y)});
}

/// Construct a hashmap for uint keys
fn uint_hash<V: copy>() -> hashmap<uint, V> {
   return hashmap(|x| { uint::hash(*x) }, |x, y| { uint::eq(*x, *y) } );
}

/// Convenience function for adding keys to a hashmap with nil type keys
fn set_add<K: const copy>(set: set<K>, +key: K) -> bool {
    set.insert(key, ())
}

/// Convert a set into a vector.
fn vec_from_set<T: copy>(s: set<T>) -> ~[T] {
    let mut v = ~[];
    vec::reserve(v, s.size());
    do s.each_key() |k| {
        vec::push(v, k);
        true
    };
    v
}

/// Construct a hashmap from a vector
fn hash_from_vec<K: const copy, V: copy>(+hasher: hashfn<K>, +eqer: eqfn<K>,
                                         items: &[(K, V)]) -> hashmap<K, V> {
    let map = hashmap(hasher, eqer);
    do vec::iter(items) |item| {
        let (key, value) = item;
        map.insert(key, value);
    }
    map
}

/// Construct a hashmap from a vector with string keys
fn hash_from_strs<V: copy>(items: &[(~str, V)]) -> hashmap<~str, V> {
    hash_from_vec(str::hash, str::eq, items)
}

/// Construct a hashmap from a vector with byte keys
fn hash_from_bytes<V: copy>(items: &[(~[u8], V)]) -> hashmap<~[u8], V> {
    hash_from_vec(vec::u8::hash, vec::u8::eq, items)
}

/// Construct a hashmap from a vector with int keys
fn hash_from_ints<V: copy>(items: &[(int, V)]) -> hashmap<int, V> {
    hash_from_vec(|x| { int::hash(*x) }, |x, y| { int::eq(*x, *y) }, items)
}

/// Construct a hashmap from a vector with uint keys
fn hash_from_uints<V: copy>(items: &[(uint, V)]) -> hashmap<uint, V> {
    hash_from_vec(|x| { uint::hash(*x) }, |x, y| { uint::eq(*x, *y) } , items)
}

// XXX Transitional
impl<K: Eq IterBytes Hash copy, V: copy> Managed<LinearMap<K, V>>:
    map<K, V> {
    pure fn size() -> uint {
        unchecked {
            do self.borrow_const |p| {
                p.len()
            }
        }
    }

    fn insert(+key: K, +value: V) -> bool {
        do self.borrow_mut |p| {
            p.insert(key, value)
        }
    }

    fn contains_key(+key: K) -> bool {
        do self.borrow_const |p| {
            p.contains_key(&key)
        }
    }

    fn contains_key_ref(key: &K) -> bool {
        do self.borrow_const |p| {
            p.contains_key(key)
        }
    }

    fn get(+key: K) -> V {
        do self.borrow_const |p| {
            p.get(&key)
        }
    }

    pure fn find(+key: K) -> Option<V> {
        unchecked {
            do self.borrow_const |p| {
                p.find(&key)
            }
        }
    }

    fn remove(+key: K) -> bool {
        do self.borrow_mut |p| {
            p.remove(&key)
        }
    }

    fn clear() {
        do self.borrow_mut |p| {
            p.clear()
        }
    }

    pure fn each(op: fn(+key: K, +value: V) -> bool) {
        unchecked {
            do self.borrow_imm |p| {
                p.each(op)
            }
        }
    }

    pure fn each_key(op: fn(+key: K) -> bool) {
        unchecked {
            do self.borrow_imm |p| {
                p.each_key(op)
            }
        }
    }

    pure fn each_value(op: fn(+value: V) -> bool) {
        unchecked {
            do self.borrow_imm |p| {
                p.each_value(op)
            }
        }
    }

    pure fn each_ref(op: fn(key: &K, value: &V) -> bool) {
        unchecked {
            do self.borrow_imm |p| {
                p.each_ref(op)
            }
        }
    }

    pure fn each_key_ref(op: fn(key: &K) -> bool) {
        unchecked {
            do self.borrow_imm |p| {
                p.each_key_ref(op)
            }
        }
    }

    pure fn each_value_ref(op: fn(value: &V) -> bool) {
        unchecked {
            do self.borrow_imm |p| {
                p.each_value_ref(op)
            }
        }
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn test_simple() {
        debug!("*** starting test_simple");
        pure fn eq_uint(x: &uint, y: &uint) -> bool { *x == *y }
        pure fn uint_id(x: &uint) -> uint { *x }
        let hasher_uint: map::hashfn<uint> = uint_id;
        let eqer_uint: map::eqfn<uint> = eq_uint;
        let hasher_str: map::hashfn<~str> = str::hash;
        let eqer_str: map::eqfn<~str> = str::eq;
        debug!("uint -> uint");
        let hm_uu: map::hashmap<uint, uint> =
            map::hashmap::<uint, uint>(copy hasher_uint, copy eqer_uint);
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
        let ten: ~str = ~"ten";
        let eleven: ~str = ~"eleven";
        let twelve: ~str = ~"twelve";
        debug!("str -> uint");
        let hm_su: map::hashmap<~str, uint> =
            map::hashmap::<~str, uint>(copy hasher_str, copy eqer_str);
        assert (hm_su.insert(~"ten", 12u));
        assert (hm_su.insert(eleven, 13u));
        assert (hm_su.insert(~"twelve", 14u));
        assert (hm_su.get(eleven) == 13u);
        assert (hm_su.get(~"eleven") == 13u);
        assert (hm_su.get(~"twelve") == 14u);
        assert (hm_su.get(~"ten") == 12u);
        assert (!hm_su.insert(~"twelve", 14u));
        assert (hm_su.get(~"twelve") == 14u);
        assert (!hm_su.insert(~"twelve", 12u));
        assert (hm_su.get(~"twelve") == 12u);
        debug!("uint -> str");
        let hm_us: map::hashmap<uint, ~str> =
            map::hashmap::<uint, ~str>(copy hasher_uint, copy eqer_uint);
        assert (hm_us.insert(10u, ~"twelve"));
        assert (hm_us.insert(11u, ~"thirteen"));
        assert (hm_us.insert(12u, ~"fourteen"));
        assert hm_us.get(11u) == ~"thirteen";
        assert hm_us.get(12u) == ~"fourteen";
        assert hm_us.get(10u) == ~"twelve";
        assert (!hm_us.insert(12u, ~"fourteen"));
        assert hm_us.get(12u) == ~"fourteen";
        assert (!hm_us.insert(12u, ~"twelve"));
        assert hm_us.get(12u) == ~"twelve";
        debug!("str -> str");
        let hm_ss: map::hashmap<~str, ~str> =
            map::hashmap::<~str, ~str>(copy hasher_str, copy eqer_str);
        assert (hm_ss.insert(ten, ~"twelve"));
        assert (hm_ss.insert(eleven, ~"thirteen"));
        assert (hm_ss.insert(twelve, ~"fourteen"));
        assert hm_ss.get(~"eleven") == ~"thirteen";
        assert hm_ss.get(~"twelve") == ~"fourteen";
        assert hm_ss.get(~"ten") == ~"twelve";
        assert (!hm_ss.insert(~"twelve", ~"fourteen"));
        assert hm_ss.get(~"twelve") == ~"fourteen";
        assert (!hm_ss.insert(~"twelve", ~"twelve"));
        assert hm_ss.get(~"twelve") == ~"twelve";
        debug!("*** finished test_simple");
    }


    /**
    * Force map growth
    */
    #[test]
    fn test_growth() {
        debug!("*** starting test_growth");
        let num_to_insert: uint = 64u;
        pure fn eq_uint(x: &uint, y: &uint) -> bool { *x == *y }
        pure fn uint_id(x: &uint) -> uint { *x }
        debug!("uint -> uint");
        let hasher_uint: map::hashfn<uint> = uint_id;
        let eqer_uint: map::eqfn<uint> = eq_uint;
        let hm_uu: map::hashmap<uint, uint> =
            map::hashmap::<uint, uint>(hasher_uint, eqer_uint);
        let mut i: uint = 0u;
        while i < num_to_insert {
            assert (hm_uu.insert(i, i * i));
            debug!("inserting %u -> %u", i, i*i);
            i += 1u;
        }
        debug!("-----");
        i = 0u;
        while i < num_to_insert {
            debug!("get(%u) = %u", i, hm_uu.get(i));
            assert (hm_uu.get(i) == i * i);
            i += 1u;
        }
        assert (hm_uu.insert(num_to_insert, 17u));
        assert (hm_uu.get(num_to_insert) == 17u);
        debug!("-----");
        i = 0u;
        while i < num_to_insert {
            debug!("get(%u) = %u", i, hm_uu.get(i));
            assert (hm_uu.get(i) == i * i);
            i += 1u;
        }
        debug!("str -> str");
        let hasher_str: map::hashfn<~str> = str::hash;
        let eqer_str: map::eqfn<~str> = str::eq;
        let hm_ss: map::hashmap<~str, ~str> =
            map::hashmap::<~str, ~str>(hasher_str, eqer_str);
        i = 0u;
        while i < num_to_insert {
            assert hm_ss.insert(uint::to_str(i, 2u), uint::to_str(i * i, 2u));
            debug!("inserting \"%s\" -> \"%s\"",
                   uint::to_str(i, 2u),
                   uint::to_str(i*i, 2u));
            i += 1u;
        }
        debug!("-----");
        i = 0u;
        while i < num_to_insert {
            debug!("get(\"%s\") = \"%s\"",
                   uint::to_str(i, 2u),
                   hm_ss.get(uint::to_str(i, 2u)));
            assert hm_ss.get(uint::to_str(i, 2u)) == uint::to_str(i * i, 2u);
            i += 1u;
        }
        assert (hm_ss.insert(uint::to_str(num_to_insert, 2u),
                             uint::to_str(17u, 2u)));
        assert hm_ss.get(uint::to_str(num_to_insert, 2u)) ==
            uint::to_str(17u, 2u);
        debug!("-----");
        i = 0u;
        while i < num_to_insert {
            debug!("get(\"%s\") = \"%s\"",
                   uint::to_str(i, 2u),
                   hm_ss.get(uint::to_str(i, 2u)));
            assert hm_ss.get(uint::to_str(i, 2u)) == uint::to_str(i * i, 2u);
            i += 1u;
        }
        debug!("*** finished test_growth");
    }

    #[test]
    fn test_removal() {
        debug!("*** starting test_removal");
        let num_to_insert: uint = 64u;
        fn eq(x: &uint, y: &uint) -> bool { *x == *y }
        fn hash(u: &uint) -> uint {
            // This hash function intentionally causes collisions between
            // consecutive integer pairs.
            *u / 2u * 2u
        }
        assert (hash(&0u) == hash(&1u));
        assert (hash(&2u) == hash(&3u));
        assert (hash(&0u) != hash(&2u));
        let hasher: map::hashfn<uint> = hash;
        let eqer: map::eqfn<uint> = eq;
        let hm: map::hashmap<uint, uint> =
            map::hashmap::<uint, uint>(hasher, eqer);
        let mut i: uint = 0u;
        while i < num_to_insert {
            assert (hm.insert(i, i * i));
            debug!("inserting %u -> %u", i, i*i);
            i += 1u;
        }
        assert (hm.size() == num_to_insert);
        debug!("-----");
        debug!("removing evens");
        i = 0u;
        while i < num_to_insert {
            let v = hm.remove(i);
            assert v;
            i += 2u;
        }
        assert (hm.size() == num_to_insert / 2u);
        debug!("-----");
        i = 1u;
        while i < num_to_insert {
            debug!("get(%u) = %u", i, hm.get(i));
            assert (hm.get(i) == i * i);
            i += 2u;
        }
        debug!("-----");
        i = 1u;
        while i < num_to_insert {
            debug!("get(%u) = %u", i, hm.get(i));
            assert (hm.get(i) == i * i);
            i += 2u;
        }
        debug!("-----");
        i = 0u;
        while i < num_to_insert {
            assert (hm.insert(i, i * i));
            debug!("inserting %u -> %u", i, i*i);
            i += 2u;
        }
        assert (hm.size() == num_to_insert);
        debug!("-----");
        i = 0u;
        while i < num_to_insert {
            debug!("get(%u) = %u", i, hm.get(i));
            assert (hm.get(i) == i * i);
            i += 1u;
        }
        debug!("-----");
        assert (hm.size() == num_to_insert);
        i = 0u;
        while i < num_to_insert {
            debug!("get(%u) = %u", i, hm.get(i));
            assert (hm.get(i) == i * i);
            i += 1u;
        }
        debug!("*** finished test_removal");
    }

    #[test]
    fn test_contains_key() {
        let key = ~"k";
        let map = map::hashmap::<~str, ~str>(str::hash, str::eq);
        assert (!map.contains_key(key));
        map.insert(key, ~"val");
        assert (map.contains_key(key));
    }

    #[test]
    fn test_find() {
        let key = ~"k";
        let map = map::hashmap::<~str, ~str>(str::hash, str::eq);
        assert (option::is_none(map.find(key)));
        map.insert(key, ~"val");
        assert (option::get(map.find(key)) == ~"val");
    }

    #[test]
    fn test_clear() {
        let key = ~"k";
        let map = map::hashmap::<~str, ~str>(str::hash, str::eq);
        map.insert(key, ~"val");
        assert (map.size() == 1);
        assert (map.contains_key(key));
        map.clear();
        assert (map.size() == 0);
        assert (!map.contains_key(key));
    }

    #[test]
    fn test_hash_from_vec() {
        let map = map::hash_from_strs(~[
            (~"a", 1),
            (~"b", 2),
            (~"c", 3)
        ]);
        assert map.size() == 3u;
        assert map.get(~"a") == 1;
        assert map.get(~"b") == 2;
        assert map.get(~"c") == 3;
    }
}
