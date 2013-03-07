// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!
 * A simple map based on a vector for small integer keys. Space requirements
 * are O(highest integer key).
 */

use core::container::{Container, Mutable, Map, Set};
use core::iter::{BaseIter, ReverseIter};
use core::option::{Some, None};
use core::prelude::*;

pub struct SmallIntMap<T> {
    priv v: ~[Option<T>],
}

impl<V> BaseIter<(uint, &self/V)> for SmallIntMap<V> {
    /// Visit all key-value pairs in order
    pure fn each(&self, it: &fn(&(uint, &self/V)) -> bool) {
        for uint::range(0, self.v.len()) |i| {
            match self.v[i] {
              Some(ref elt) => if !it(&(i, elt)) { break },
              None => ()
            }
        }
    }

    pure fn size_hint(&self) -> Option<uint> { Some(self.len()) }
}

impl<V> ReverseIter<(uint, &self/V)> for SmallIntMap<V> {
    /// Visit all key-value pairs in reverse order
    pure fn each_reverse(&self, it: &fn(&(uint, &self/V)) -> bool) {
        for uint::range_rev(self.v.len(), 0) |i| {
            match self.v[i - 1] {
              Some(ref elt) => if !it(&(i - 1, elt)) { break },
              None => ()
            }
        }
    }
}

impl<V> Container for SmallIntMap<V> {
    /// Return the number of elements in the map
    pure fn len(&self) -> uint {
        let mut sz = 0;
        for self.v.each |item| {
            if item.is_some() {
                sz += 1;
            }
        }
        sz
    }

    /// Return true if the map contains no elements
    pure fn is_empty(&self) -> bool { self.len() == 0 }
}

impl<V> Mutable for SmallIntMap<V> {
    /// Clear the map, removing all key-value pairs.
    fn clear(&mut self) { self.v.clear() }
}

impl<V> Map<uint, V> for SmallIntMap<V> {
    /// Return true if the map contains a value for the specified key
    pure fn contains_key(&self, key: &uint) -> bool {
        self.find(key).is_some()
    }

    /// Visit all keys in order
    pure fn each_key(&self, blk: &fn(key: &uint) -> bool) {
        self.each(|&(k, _)| blk(&k))
    }

    /// Visit all values in order
    pure fn each_value(&self, blk: &fn(value: &V) -> bool) {
        self.each(|&(_, v)| blk(v))
    }

    /// Return the value corresponding to the key in the map
    pure fn find(&self, key: &uint) -> Option<&self/V> {
        if *key < self.v.len() {
            match self.v[*key] {
              Some(ref value) => Some(value),
              None => None
            }
        } else {
            None
        }
    }

    /// Insert a key-value pair into the map. An existing value for a
    /// key is replaced by the new value. Return true if the key did
    /// not already exist in the map.
    fn insert(&mut self, key: uint, value: V) -> bool {
        let exists = self.contains_key(&key);
        let len = self.v.len();
        if len <= key {
            vec::grow_fn(&mut self.v, key - len + 1, |_| None);
        }
        self.v[key] = Some(value);
        !exists
    }

    /// Remove a key-value pair from the map. Return true if the key
    /// was present in the map, otherwise false.
    fn remove(&mut self, key: &uint) -> bool {
        if *key >= self.v.len() {
            return false;
        }
        let removed = self.v[*key].is_some();
        self.v[*key] = None;
        removed
    }
}

pub impl<V> SmallIntMap<V> {
    /// Create an empty SmallIntMap
    static pure fn new() -> SmallIntMap<V> { SmallIntMap{v: ~[]} }

    pure fn get(&self, key: &uint) -> &self/V {
        self.find(key).expect("key not present")
    }
}

pub impl<V:Copy> SmallIntMap<V> {
    fn update_with_key(&mut self, key: uint, val: V,
                       ff: &fn(uint, V, V) -> V) -> bool {
        let new_val = match self.find(&key) {
            None => val,
            Some(orig) => ff(key, *orig, val)
        };
        self.insert(key, new_val)
    }

    fn update(&mut self, key: uint, newval: V, ff: &fn(V, V) -> V) -> bool {
        self.update_with_key(key, newval, |_k, v, v1| ff(v,v1))
    }
}

#[cfg(test)]
mod tests {
    use super::SmallIntMap;

    #[test]
    fn test_len() {
        let mut map = SmallIntMap::new();
        fail_unless!(map.len() == 0);
        fail_unless!(map.is_empty());
        fail_unless!(map.insert(5, 20));
        fail_unless!(map.len() == 1);
        fail_unless!(!map.is_empty());
        fail_unless!(map.insert(11, 12));
        fail_unless!(map.len() == 2);
        fail_unless!(!map.is_empty());
        fail_unless!(map.insert(14, 22));
        fail_unless!(map.len() == 3);
        fail_unless!(!map.is_empty());
    }

    #[test]
    fn test_clear() {
        let mut map = SmallIntMap::new();
        fail_unless!(map.insert(5, 20));
        fail_unless!(map.insert(11, 12));
        fail_unless!(map.insert(14, 22));
        map.clear();
        fail_unless!(map.is_empty());
        fail_unless!(map.find(&5).is_none());
        fail_unless!(map.find(&11).is_none());
        fail_unless!(map.find(&14).is_none());
    }

    #[test]
    fn test_insert_with_key() {
        let mut map = SmallIntMap::new();

        // given a new key, initialize it with this new count, given
        // given an existing key, add more to its count
        fn addMoreToCount(_k: uint, v0: uint, v1: uint) -> uint {
            v0 + v1
        }

        fn addMoreToCount_simple(v0: uint, v1: uint) -> uint {
            v0 + v1
        }

        // count integers
        map.update(3, 1, addMoreToCount_simple);
        map.update_with_key(9, 1, addMoreToCount);
        map.update(3, 7, addMoreToCount_simple);
        map.update_with_key(5, 3, addMoreToCount);
        map.update_with_key(3, 2, addMoreToCount);

        // check the total counts
        fail_unless!(map.find(&3).get() == &10);
        fail_unless!(map.find(&5).get() == &3);
        fail_unless!(map.find(&9).get() == &1);

        // sadly, no sevens were counted
        fail_unless!(map.find(&7).is_none());
    }
}
