// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! A deprecated compatibility layer on top of `core::hashmap`

use core::prelude::*;
use core::hash::Hash;
use core::prelude::*;
use core::to_bytes::IterBytes;
use core::vec;

/// A convenience type to treat a hashmap as a set
pub type Set<K> = HashMap<K, ()>;

pub type HashMap<K, V> = chained::T<K, V>;

pub mod chained {
    use core::ops;
    use core::prelude::*;
    use core::hashmap::linear::LinearMap;

    struct HashMap_<K, V> {
        priv map: LinearMap<K, V>
    }

    pub type T<K, V> = @mut HashMap_<K, V>;

    pub impl<K:Eq + IterBytes + Hash,V> HashMap_<K, V> {
        fn clear(&mut self) {
            self.map.clear()
        }
    }

    impl<K:Eq + IterBytes + Hash,V> Container for HashMap_<K, V> {
        fn len(&const self) -> uint { self.map.len() }
        fn is_empty(&const self) -> bool { self.map.is_empty() }
    }

    pub impl<K:Eq + IterBytes + Hash,V> HashMap_<K, V> {
        fn contains_key(&self, k: &K) -> bool {
            self.map.contains_key(k)
        }

        fn insert(&mut self, k: K, v: V) -> bool {
            self.map.insert(k, v)
        }

        fn remove(&mut self, k: &K) -> bool {
            self.map.remove(k)
        }

        fn each(&self, blk: &fn(key: &K, value: &V) -> bool) {
            do self.map.each |&(k, v)| { blk(k, v) }
        }

        fn each_key(&self, blk: &fn(key: &K) -> bool) {
            self.map.each_key(blk)
        }

        fn each_value(&self, blk: &fn(value: &V) -> bool) {
            self.map.each_value(blk)
        }
    }

    pub impl<K:Eq + IterBytes + Hash + Copy,V:Copy> HashMap_<K, V> {
        fn find(&self, k: &K) -> Option<V> {
            self.map.find(k).map(|&x| copy *x)
        }

        fn update(&mut self, key: K, newval: V, ff: &fn(V, V) -> V) -> bool {
            match self.find(&key) {
                None => self.insert(key, newval),
                Some(orig) => self.insert(key, ff(orig, newval))
            }
        }

        fn get(&self, k: &K) -> V {
            copy *self.map.get(k)
        }
    }

    impl<K:Eq + IterBytes + Hash + Copy,V:Copy> ops::Index<K, V>
            for HashMap_<K, V> {
        fn index(&self, k: K) -> V {
            self.get(&k)
        }
    }

    pub fn mk<K:Eq + IterBytes + Hash,V:Copy>() -> T<K,V> {
        @mut HashMap_{map: LinearMap::new()}
    }
}

/*
Function: hashmap

Construct a hashmap.
*/
pub fn HashMap<K:Eq + IterBytes + Hash + Const,V:Copy>()
        -> HashMap<K, V> {
    chained::mk()
}

/// Convenience function for adding keys to a hashmap with nil type keys
pub fn set_add<K:Eq + IterBytes + Hash + Const + Copy>(set: Set<K>, key: K)
                                                    -> bool {
    set.insert(key, ())
}

/// Convert a set into a vector.
pub fn vec_from_set<T:Eq + IterBytes + Hash + Copy>(s: Set<T>) -> ~[T] {
    do vec::build_sized(s.len()) |push| {
        for s.each_key() |&k| {
            push(k);
        }
    }
}

/// Construct a hashmap from a vector
pub fn hash_from_vec<K:Eq + IterBytes + Hash + Const + Copy,V:Copy>(
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

#[cfg(test)]
mod tests {
    use core::uint;

    use super::*;

    #[test]
    fn test_simple() {
        debug!("*** starting test_simple");
        fn eq_uint(x: &uint, y: &uint) -> bool { *x == *y }
        fn uint_id(x: &uint) -> uint { *x }
        debug!("uint -> uint");
        let hm_uu: HashMap<uint, uint> =
            HashMap::<uint, uint>();
        fail_unless!((hm_uu.insert(10u, 12u)));
        fail_unless!((hm_uu.insert(11u, 13u)));
        fail_unless!((hm_uu.insert(12u, 14u)));
        fail_unless!((hm_uu.get(&11) == 13u));
        fail_unless!((hm_uu.get(&12) == 14u));
        fail_unless!((hm_uu.get(&10) == 12u));
        fail_unless!((!hm_uu.insert(12u, 14u)));
        fail_unless!((hm_uu.get(&12) == 14u));
        fail_unless!((!hm_uu.insert(12u, 12u)));
        fail_unless!((hm_uu.get(&12) == 12u));
        let ten: ~str = ~"ten";
        let eleven: ~str = ~"eleven";
        let twelve: ~str = ~"twelve";
        debug!("str -> uint");
        let hm_su: HashMap<~str, uint> =
            HashMap::<~str, uint>();
        fail_unless!((hm_su.insert(~"ten", 12u)));
        fail_unless!((hm_su.insert(eleven, 13u)));
        fail_unless!((hm_su.insert(~"twelve", 14u)));
        fail_unless!((hm_su.get(&eleven) == 13u));
        fail_unless!((hm_su.get(&~"eleven") == 13u));
        fail_unless!((hm_su.get(&~"twelve") == 14u));
        fail_unless!((hm_su.get(&~"ten") == 12u));
        fail_unless!((!hm_su.insert(~"twelve", 14u)));
        fail_unless!((hm_su.get(&~"twelve") == 14u));
        fail_unless!((!hm_su.insert(~"twelve", 12u)));
        fail_unless!((hm_su.get(&~"twelve") == 12u));
        debug!("uint -> str");
        let hm_us: HashMap<uint, ~str> =
            HashMap::<uint, ~str>();
        fail_unless!((hm_us.insert(10u, ~"twelve")));
        fail_unless!((hm_us.insert(11u, ~"thirteen")));
        fail_unless!((hm_us.insert(12u, ~"fourteen")));
        fail_unless!(hm_us.get(&11) == ~"thirteen");
        fail_unless!(hm_us.get(&12) == ~"fourteen");
        fail_unless!(hm_us.get(&10) == ~"twelve");
        fail_unless!((!hm_us.insert(12u, ~"fourteen")));
        fail_unless!(hm_us.get(&12) == ~"fourteen");
        fail_unless!((!hm_us.insert(12u, ~"twelve")));
        fail_unless!(hm_us.get(&12) == ~"twelve");
        debug!("str -> str");
        let hm_ss: HashMap<~str, ~str> =
            HashMap::<~str, ~str>();
        fail_unless!((hm_ss.insert(ten, ~"twelve")));
        fail_unless!((hm_ss.insert(eleven, ~"thirteen")));
        fail_unless!((hm_ss.insert(twelve, ~"fourteen")));
        fail_unless!(hm_ss.get(&~"eleven") == ~"thirteen");
        fail_unless!(hm_ss.get(&~"twelve") == ~"fourteen");
        fail_unless!(hm_ss.get(&~"ten") == ~"twelve");
        fail_unless!((!hm_ss.insert(~"twelve", ~"fourteen")));
        fail_unless!(hm_ss.get(&~"twelve") == ~"fourteen");
        fail_unless!((!hm_ss.insert(~"twelve", ~"twelve")));
        fail_unless!(hm_ss.get(&~"twelve") == ~"twelve");
        debug!("*** finished test_simple");
    }


    /**
    * Force map growth
    */
    #[test]
    fn test_growth() {
        debug!("*** starting test_growth");
        let num_to_insert: uint = 64u;
        fn eq_uint(x: &uint, y: &uint) -> bool { *x == *y }
        fn uint_id(x: &uint) -> uint { *x }
        debug!("uint -> uint");
        let hm_uu: HashMap<uint, uint> =
            HashMap::<uint, uint>();
        let mut i: uint = 0u;
        while i < num_to_insert {
            fail_unless!((hm_uu.insert(i, i * i)));
            debug!("inserting %u -> %u", i, i*i);
            i += 1u;
        }
        debug!("-----");
        i = 0u;
        while i < num_to_insert {
            debug!("get(%u) = %u", i, hm_uu.get(&i));
            fail_unless!((hm_uu.get(&i) == i * i));
            i += 1u;
        }
        fail_unless!((hm_uu.insert(num_to_insert, 17u)));
        fail_unless!((hm_uu.get(&num_to_insert) == 17u));
        debug!("-----");
        i = 0u;
        while i < num_to_insert {
            debug!("get(%u) = %u", i, hm_uu.get(&i));
            fail_unless!((hm_uu.get(&i) == i * i));
            i += 1u;
        }
        debug!("str -> str");
        let hm_ss: HashMap<~str, ~str> =
            HashMap::<~str, ~str>();
        i = 0u;
        while i < num_to_insert {
            fail_unless!(hm_ss.insert(uint::to_str_radix(i, 2u),
                                uint::to_str_radix(i * i, 2u)));
            debug!("inserting \"%s\" -> \"%s\"",
                   uint::to_str_radix(i, 2u),
                   uint::to_str_radix(i*i, 2u));
            i += 1u;
        }
        debug!("-----");
        i = 0u;
        while i < num_to_insert {
            debug!("get(\"%s\") = \"%s\"",
                   uint::to_str_radix(i, 2u),
                   hm_ss.get(&uint::to_str_radix(i, 2u)));
            fail_unless!(hm_ss.get(&uint::to_str_radix(i, 2u)) ==
                             uint::to_str_radix(i * i, 2u));
            i += 1u;
        }
        fail_unless!(hm_ss.insert(uint::to_str_radix(num_to_insert, 2u),
                             uint::to_str_radix(17u, 2u)));
        fail_unless!(hm_ss.get(&uint::to_str_radix(num_to_insert, 2u)) ==
            uint::to_str_radix(17u, 2u));
        debug!("-----");
        i = 0u;
        while i < num_to_insert {
            debug!("get(\"%s\") = \"%s\"",
                   uint::to_str_radix(i, 2u),
                   hm_ss.get(&uint::to_str_radix(i, 2u)));
            fail_unless!(hm_ss.get(&uint::to_str_radix(i, 2u)) ==
                             uint::to_str_radix(i * i, 2u));
            i += 1u;
        }
        debug!("*** finished test_growth");
    }

    #[test]
    fn test_removal() {
        debug!("*** starting test_removal");
        let num_to_insert: uint = 64u;
        let hm: HashMap<uint, uint> =
            HashMap::<uint, uint>();
        let mut i: uint = 0u;
        while i < num_to_insert {
            fail_unless!((hm.insert(i, i * i)));
            debug!("inserting %u -> %u", i, i*i);
            i += 1u;
        }
        fail_unless!((hm.len() == num_to_insert));
        debug!("-----");
        debug!("removing evens");
        i = 0u;
        while i < num_to_insert {
            let v = hm.remove(&i);
            fail_unless!(v);
            i += 2u;
        }
        fail_unless!((hm.len() == num_to_insert / 2u));
        debug!("-----");
        i = 1u;
        while i < num_to_insert {
            debug!("get(%u) = %u", i, hm.get(&i));
            fail_unless!((hm.get(&i) == i * i));
            i += 2u;
        }
        debug!("-----");
        i = 1u;
        while i < num_to_insert {
            debug!("get(%u) = %u", i, hm.get(&i));
            fail_unless!((hm.get(&i) == i * i));
            i += 2u;
        }
        debug!("-----");
        i = 0u;
        while i < num_to_insert {
            fail_unless!((hm.insert(i, i * i)));
            debug!("inserting %u -> %u", i, i*i);
            i += 2u;
        }
        fail_unless!((hm.len() == num_to_insert));
        debug!("-----");
        i = 0u;
        while i < num_to_insert {
            debug!("get(%u) = %u", i, hm.get(&i));
            fail_unless!((hm.get(&i) == i * i));
            i += 1u;
        }
        debug!("-----");
        fail_unless!((hm.len() == num_to_insert));
        i = 0u;
        while i < num_to_insert {
            debug!("get(%u) = %u", i, hm.get(&i));
            fail_unless!((hm.get(&i) == i * i));
            i += 1u;
        }
        debug!("*** finished test_removal");
    }

    #[test]
    fn test_contains_key() {
        let key = ~"k";
        let map = HashMap::<~str, ~str>();
        fail_unless!((!map.contains_key(&key)));
        map.insert(key, ~"val");
        fail_unless!((map.contains_key(&key)));
    }

    #[test]
    fn test_find() {
        let key = ~"k";
        let map = HashMap::<~str, ~str>();
        fail_unless!(map.find(&key).is_none());
        map.insert(key, ~"val");
        fail_unless!(map.find(&key).get() == ~"val");
    }

    #[test]
    fn test_clear() {
        let key = ~"k";
        let mut map = HashMap::<~str, ~str>();
        map.insert(key, ~"val");
        fail_unless!((map.len() == 1));
        fail_unless!((map.contains_key(&key)));
        map.clear();
        fail_unless!((map.len() == 0));
        fail_unless!((!map.contains_key(&key)));
    }

    #[test]
    fn test_hash_from_vec() {
        let map = hash_from_vec(~[
            (~"a", 1),
            (~"b", 2),
            (~"c", 3)
        ]);
        fail_unless!(map.len() == 3u);
        fail_unless!(map.get(&~"a") == 1);
        fail_unless!(map.get(&~"b") == 2);
        fail_unless!(map.get(&~"c") == 3);
    }
}
