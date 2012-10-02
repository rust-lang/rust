//! A map type

#[forbid(deprecated_mode)];

use io::WriterUtil;
use to_str::ToStr;
use mutable::Mut;
use send_map::linear::LinearMap;

use core::cmp::Eq;
use hash::Hash;
use to_bytes::IterBytes;

/// A convenience type to treat a hashmap as a set
pub type Set<K:Eq IterBytes Hash> = HashMap<K, ()>;

pub type HashMap<K:Eq IterBytes Hash, V> = chained::T<K, V>;

pub trait Map<K:Eq IterBytes Hash Copy, V: Copy> {
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
    fn insert(+v: K, +v: V) -> bool;

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
    pub type Rational = {num: int, den: int}; // : int::positive(*.den);

    pub pure fn rational_leq(x: Rational, y: Rational) -> bool {
        // NB: Uses the fact that rationals have positive denominators WLOG:

        x.num * y.den <= y.num * x.den
    }
}


// FIXME (#2344): package this up and export it as a datatype usable for
// external code that doesn't want to pay the cost of a box.
pub mod chained {

    const initial_capacity: uint = 32u; // 2^5

    struct Entry<K, V> {
        hash: uint,
        key: K,
        value: V,
        mut next: Option<@Entry<K, V>>
    }

    struct HashMap_<K:Eq IterBytes Hash, V> {
        mut count: uint,
        mut chains: ~[mut Option<@Entry<K,V>>]
    }

    pub type T<K:Eq IterBytes Hash, V> = @HashMap_<K, V>;

    enum SearchResult<K, V> {
        NotFound,
        FoundFirst(uint, @Entry<K,V>),
        FoundAfter(@Entry<K,V>, @Entry<K,V>)
    }

    priv impl<K:Eq IterBytes Hash, V: Copy> T<K, V> {
        pure fn search_rem(k: &K, h: uint, idx: uint,
                           e_root: @Entry<K,V>) -> SearchResult<K,V> {
            let mut e0 = e_root;
            let mut comp = 1u;   // for logging
            loop {
                match copy e0.next {
                  None => {
                    debug!("search_tbl: absent, comp %u, hash %u, idx %u",
                           comp, h, idx);
                    return NotFound;
                  }
                  Some(e1) => {
                    comp += 1u;
                    unsafe {
                        if e1.hash == h && e1.key == *k {
                            debug!("search_tbl: present, comp %u, \
                                    hash %u, idx %u",
                                   comp, h, idx);
                            return FoundAfter(e0, e1);
                        } else {
                            e0 = e1;
                        }
                    }
                  }
                }
            };
        }

        pure fn search_tbl(k: &K, h: uint) -> SearchResult<K,V> {
            let idx = h % vec::len(self.chains);
            match copy self.chains[idx] {
              None => {
                debug!("search_tbl: none, comp %u, hash %u, idx %u",
                       0u, h, idx);
                return NotFound;
              }
              Some(e) => {
                unsafe {
                    if e.hash == h && e.key == *k {
                        debug!("search_tbl: present, comp %u, hash %u, \
                                idx %u", 1u, h, idx);
                        return FoundFirst(idx, e);
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
            self.chains <- new_chains;
        }

        pure fn each_entry(blk: fn(@Entry<K,V>) -> bool) {
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

    impl<K:Eq IterBytes Hash Copy, V: Copy> T<K, V>: Map<K, V> {
        pure fn size() -> uint { self.count }

        fn contains_key(+k: K) -> bool {
            self.contains_key_ref(&k)
        }

        fn contains_key_ref(k: &K) -> bool {
            let hash = k.hash_keyed(0,0) as uint;
            match self.search_tbl(k, hash) {
              NotFound => false,
              FoundFirst(*) | FoundAfter(*) => true
            }
        }

        fn insert(+k: K, +v: V) -> bool {
            let hash = k.hash_keyed(0,0) as uint;
            match self.search_tbl(&k, hash) {
              NotFound => {
                self.count += 1u;
                let idx = hash % vec::len(self.chains);
                let old_chain = self.chains[idx];
                self.chains[idx] = Some(@Entry {
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
              FoundFirst(idx, entry) => {
                self.chains[idx] = Some(@Entry {
                    hash: hash,
                    key: k,
                    value: v,
                    next: entry.next});
                return false;
              }
              FoundAfter(prev, entry) => {
                prev.next = Some(@Entry {
                    hash: hash,
                    key: k,
                    value: v,
                    next: entry.next});
                return false;
              }
            }
        }

        pure fn find(+k: K) -> Option<V> {
            unsafe {
                match self.search_tbl(&k, k.hash_keyed(0,0) as uint) {
                  NotFound => None,
                  FoundFirst(_, entry) => Some(entry.value),
                  FoundAfter(_, entry) => Some(entry.value)
                }
            }
        }

        fn get(+k: K) -> V {
            let opt_v = self.find(k);
            if opt_v.is_none() {
                fail fmt!("Key not found in table: %?", k);
            }
            option::unwrap(move opt_v)
        }

        fn remove(+k: K) -> bool {
            match self.search_tbl(&k, k.hash_keyed(0,0) as uint) {
              NotFound => false,
              FoundFirst(idx, entry) => {
                self.count -= 1u;
                self.chains[idx] = entry.next;
                true
              }
              FoundAfter(eprev, entry) => {
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

    impl<K:Eq IterBytes Hash Copy ToStr, V: ToStr Copy> T<K, V>: ToStr {
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

    impl<K:Eq IterBytes Hash Copy, V: Copy> T<K, V>: ops::Index<K, V> {
        pure fn index(+k: K) -> V {
            unsafe {
                self.get(k)
            }
        }
    }

    fn chains<K,V>(nchains: uint) -> ~[mut Option<@Entry<K,V>>] {
        vec::to_mut(vec::from_elem(nchains, None))
    }

    pub fn mk<K:Eq IterBytes Hash, V: Copy>() -> T<K,V> {
        let slf: T<K, V> = @HashMap_ {count: 0u,
                                      chains: chains(initial_capacity)};
        slf
    }
}

/*
Function: hashmap

Construct a hashmap.
*/
pub fn HashMap<K:Eq IterBytes Hash Const, V: Copy>()
        -> HashMap<K, V> {
    chained::mk()
}

/// Convenience function for adding keys to a hashmap with nil type keys
pub fn set_add<K:Eq IterBytes Hash Const Copy>(set: Set<K>, +key: K) -> bool {
    set.insert(key, ())
}

/// Convert a set into a vector.
pub fn vec_from_set<T:Eq IterBytes Hash Copy>(s: Set<T>) -> ~[T] {
    do vec::build_sized(s.size()) |push| {
        for s.each_key() |k| {
            push(k);
        }
    }
}

/// Construct a hashmap from a vector
pub fn hash_from_vec<K: Eq IterBytes Hash Const Copy, V: Copy>(
    items: &[(K, V)]) -> HashMap<K, V> {
    let map = HashMap();
    for vec::each(items) |item| {
        match *item {
            (copy key, copy value) => {
                map.insert(key, value);
            }
        }
    }
    map
}

// XXX Transitional
impl<K: Eq IterBytes Hash Copy, V: Copy> @Mut<LinearMap<K, V>>:
    Map<K, V> {
    pure fn size() -> uint {
        unsafe {
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
        unsafe {
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
        unsafe {
            do self.borrow_imm |p| {
                p.each(|k, v| op(*k, *v))
            }
        }
    }

    pure fn each_key(op: fn(+key: K) -> bool) {
        unsafe {
            do self.borrow_imm |p| {
                p.each_key(|k| op(*k))
            }
        }
    }

    pure fn each_value(op: fn(+value: V) -> bool) {
        unsafe {
            do self.borrow_imm |p| {
                p.each_value(|v| op(*v))
            }
        }
    }

    pure fn each_ref(op: fn(key: &K, value: &V) -> bool) {
        unsafe {
            do self.borrow_imm |p| {
                p.each(op)
            }
        }
    }

    pure fn each_key_ref(op: fn(key: &K) -> bool) {
        unsafe {
            do self.borrow_imm |p| {
                p.each_key(op)
            }
        }
    }

    pure fn each_value_ref(op: fn(value: &V) -> bool) {
        unsafe {
            do self.borrow_imm |p| {
                p.each_value(op)
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
        debug!("uint -> uint");
        let hm_uu: map::HashMap<uint, uint> =
            map::HashMap::<uint, uint>();
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
        let hm_su: map::HashMap<~str, uint> =
            map::HashMap::<~str, uint>();
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
        let hm_us: map::HashMap<uint, ~str> =
            map::HashMap::<uint, ~str>();
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
        let hm_ss: map::HashMap<~str, ~str> =
            map::HashMap::<~str, ~str>();
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
        let hm_uu: map::HashMap<uint, uint> =
            map::HashMap::<uint, uint>();
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
        let hm_ss: map::HashMap<~str, ~str> =
            map::HashMap::<~str, ~str>();
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
        let hm: map::HashMap<uint, uint> =
            map::HashMap::<uint, uint>();
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
        let map = map::HashMap::<~str, ~str>();
        assert (!map.contains_key(key));
        map.insert(key, ~"val");
        assert (map.contains_key(key));
    }

    #[test]
    fn test_find() {
        let key = ~"k";
        let map = map::HashMap::<~str, ~str>();
        assert (option::is_none(&map.find(key)));
        map.insert(key, ~"val");
        assert (option::get(&map.find(key)) == ~"val");
    }

    #[test]
    fn test_clear() {
        let key = ~"k";
        let map = map::HashMap::<~str, ~str>();
        map.insert(key, ~"val");
        assert (map.size() == 1);
        assert (map.contains_key(key));
        map.clear();
        assert (map.size() == 0);
        assert (!map.contains_key(key));
    }

    #[test]
    fn test_hash_from_vec() {
        let map = map::hash_from_vec(~[
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
