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
#[forbid(deprecated_mode)];

use core::container::{Container, Mutable, Map, Set};
use core::option::{Some, None};
use core::prelude::*;

pub struct SmallIntMap<T> {
    priv v: ~[Option<T>],
}

impl<V> SmallIntMap<V>: Container {
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

impl<V> SmallIntMap<V>: Mutable {
    /// Clear the map, removing all key-value pairs.
    fn clear(&mut self) { self.v.clear() }
}

impl<V> SmallIntMap<V>: Map<uint, V> {
    /// Return true if the map contains a value for the specified key
    pure fn contains_key(&self, key: &uint) -> bool {
        self.find(key).is_some()
    }

    /// Visit all key-value pairs
    pure fn each(&self, it: fn(key: &uint, value: &V) -> bool) {
        for uint::range(0, self.v.len()) |i| {
            match self.v[i] {
              Some(ref elt) => if !it(&i, elt) { break },
              None => ()
            }
        }
    }

    /// Visit all keys
    pure fn each_key(&self, blk: fn(key: &uint) -> bool) {
        self.each(|k, _| blk(k))
    }

    /// Visit all values
    pure fn each_value(&self, blk: fn(value: &V) -> bool) {
        self.each(|_, v| blk(v))
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

pub impl<V: Copy> SmallIntMap<V> {
    fn update_with_key(&mut self, key: uint, val: V,
                       ff: fn(uint, V, V) -> V) -> bool {
        match self.find(&key) {
          None => self.insert(key, val),
          Some(orig) => self.insert(key, ff(key, copy *orig, val)),
        }
    }

    fn update(&mut self, key: uint, newval: V, ff: fn(V, V) -> V) -> bool {
        self.update_with_key(key, newval, |_k, v, v1| ff(v,v1))
    }
}

#[cfg(test)]
mod tests {
    use super::SmallIntMap;

    #[test]
    fn test_len() {
        let mut map = SmallIntMap::new();
        assert map.len() == 0;
        assert map.is_empty();
        assert map.insert(5, 20);
        assert map.len() == 1;
        assert !map.is_empty();
        assert map.insert(11, 12);
        assert map.len() == 2;
        assert !map.is_empty();
        assert map.insert(14, 22);
        assert map.len() == 3;
        assert !map.is_empty();
    }

    #[test]
    fn test_clear() {
        let mut map = SmallIntMap::new();
        assert map.insert(5, 20);
        assert map.insert(11, 12);
        assert map.insert(14, 22);
        map.clear();
        assert map.is_empty();
        assert map.find(&5).is_none();
        assert map.find(&11).is_none();
        assert map.find(&14).is_none();
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
        assert map.find(&3).get() == &10;
        assert map.find(&5).get() == &3;
        assert map.find(&9).get() == &1;

        // sadly, no sevens were counted
        assert map.find(&7).is_none();
    }
}
