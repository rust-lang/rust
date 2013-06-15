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

#[allow(missing_doc)];

use core::prelude::*;

use core::cmp;
use core::container::{Container, Mutable, Map, Set};
use core::old_iter::BaseIter;
use core::old_iter;
use core::uint;
use core::util::replace;
use core::vec;

#[allow(missing_doc)]
pub struct SmallIntMap<T> {
    priv v: ~[Option<T>],
}

impl<V> Container for SmallIntMap<V> {
    /// Return the number of elements in the map
    fn len(&self) -> uint {
        let mut sz = 0;
        for uint::range(0, self.v.len()) |i| {
            match self.v[i] {
                Some(_) => sz += 1,
                None => {}
            }
        }
        sz
    }

    /// Return true if the map contains no elements
    fn is_empty(&self) -> bool { self.len() == 0 }
}

impl<V> Mutable for SmallIntMap<V> {
    /// Clear the map, removing all key-value pairs.
    fn clear(&mut self) { self.v.clear() }
}

impl<V> Map<uint, V> for SmallIntMap<V> {
    /// Return true if the map contains a value for the specified key
    fn contains_key(&self, key: &uint) -> bool {
        self.find(key).is_some()
    }

    /// Visit all key-value pairs in order
    fn each<'a>(&'a self, it: &fn(&uint, &'a V) -> bool) -> bool {
        for uint::range(0, self.v.len()) |i| {
            match self.v[i] {
              Some(ref elt) => if !it(&i, elt) { return false; },
              None => ()
            }
        }
        return true;
    }

    /// Visit all keys in order
    fn each_key(&self, blk: &fn(key: &uint) -> bool) -> bool {
        self.each(|k, _| blk(k))
    }

    /// Visit all values in order
    fn each_value<'a>(&'a self, blk: &fn(value: &'a V) -> bool) -> bool {
        self.each(|_, v| blk(v))
    }

    /// Iterate over the map and mutate the contained values
    fn mutate_values(&mut self, it: &fn(&uint, &mut V) -> bool) -> bool {
        for uint::range(0, self.v.len()) |i| {
            match self.v[i] {
              Some(ref mut elt) => if !it(&i, elt) { return false; },
              None => ()
            }
        }
        return true;
    }

    /// Return a reference to the value corresponding to the key
    fn find<'a>(&'a self, key: &uint) -> Option<&'a V> {
        if *key < self.v.len() {
            match self.v[*key] {
              Some(ref value) => Some(value),
              None => None
            }
        } else {
            None
        }
    }

    /// Return a mutable reference to the value corresponding to the key
    fn find_mut<'a>(&'a mut self, key: &uint) -> Option<&'a mut V> {
        if *key < self.v.len() {
            match self.v[*key] {
              Some(ref mut value) => Some(value),
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
        self.pop(key).is_some()
    }

    /// Insert a key-value pair from the map. If the key already had a value
    /// present in the map, that value is returned. Otherwise None is returned.
    fn swap(&mut self, key: uint, value: V) -> Option<V> {
        match self.find_mut(&key) {
            Some(loc) => { return Some(replace(loc, value)); }
            None => ()
        }
        self.insert(key, value);
        return None;
    }

    /// Removes a key from the map, returning the value at the key if the key
    /// was previously in the map.
    fn pop(&mut self, key: &uint) -> Option<V> {
        if *key >= self.v.len() {
            return None;
        }
        replace(&mut self.v[*key], None)
    }
}

impl<V> SmallIntMap<V> {
    /// Create an empty SmallIntMap
    pub fn new() -> SmallIntMap<V> { SmallIntMap{v: ~[]} }

    /// Visit all key-value pairs in reverse order
    pub fn each_reverse<'a>(&'a self, it: &fn(uint, &'a V) -> bool) -> bool {
        for uint::range_rev(self.v.len(), 0) |i| {
            match self.v[i - 1] {
              Some(ref elt) => if !it(i - 1, elt) { return false; },
              None => ()
            }
        }
        return true;
    }

    pub fn get<'a>(&'a self, key: &uint) -> &'a V {
        self.find(key).expect("key not present")
    }
}

impl<V:Copy> SmallIntMap<V> {
    pub fn update_with_key(&mut self, key: uint, val: V,
                           ff: &fn(uint, V, V) -> V) -> bool {
        let new_val = match self.find(&key) {
            None => val,
            Some(orig) => ff(key, *orig, val)
        };
        self.insert(key, new_val)
    }

    pub fn update(&mut self, key: uint, newval: V, ff: &fn(V, V) -> V)
                  -> bool {
        self.update_with_key(key, newval, |_k, v, v1| ff(v,v1))
    }
}

/// A set implemented on top of the SmallIntMap type. This set is always a set
/// of integers, and the space requirements are on the order of the highest
/// valued integer in the set.
pub struct SmallIntSet {
    priv map: SmallIntMap<()>
}

impl Container for SmallIntSet {
    /// Return the number of elements in the map
    fn len(&self) -> uint {
        self.map.len()
    }

    /// Return true if the map contains no elements
    fn is_empty(&self) -> bool { self.len() == 0 }
}

impl Mutable for SmallIntSet {
    /// Clear the map, removing all key-value pairs.
    fn clear(&mut self) { self.map.clear() }
}

impl BaseIter<uint> for SmallIntSet {
    /// Visit all values in order
    fn each(&self, f: &fn(&uint) -> bool) -> bool { self.map.each_key(f) }
    fn size_hint(&self) -> Option<uint> { Some(self.len()) }
}

impl Set<uint> for SmallIntSet {
    /// Return true if the set contains a value
    fn contains(&self, value: &uint) -> bool { self.map.contains_key(value) }

    /// Add a value to the set. Return true if the value was not already
    /// present in the set.
    fn insert(&mut self, value: uint) -> bool { self.map.insert(value, ()) }

    /// Remove a value from the set. Return true if the value was
    /// present in the set.
    fn remove(&mut self, value: &uint) -> bool { self.map.remove(value) }

    /// Return true if the set has no elements in common with `other`.
    /// This is equivalent to checking for an empty uintersection.
    fn is_disjoint(&self, other: &SmallIntSet) -> bool {
        old_iter::all(self, |v| !other.contains(v))
    }

    /// Return true if the set is a subset of another
    fn is_subset(&self, other: &SmallIntSet) -> bool {
        old_iter::all(self, |v| other.contains(v))
    }

    /// Return true if the set is a superset of another
    fn is_superset(&self, other: &SmallIntSet) -> bool {
        other.is_subset(self)
    }

    /// Visit the values representing the difference
    fn difference(&self, other: &SmallIntSet, f: &fn(&uint) -> bool) -> bool {
        self.each(|v| other.contains(v) || f(v))
    }

    /// Visit the values representing the symmetric difference
    fn symmetric_difference(&self,
                            other: &SmallIntSet,
                            f: &fn(&uint) -> bool) -> bool {
        let len = cmp::max(self.map.v.len() ,other.map.v.len());

        for uint::range(0, len) |i| {
            if self.contains(&i) ^ other.contains(&i) {
                if !f(&i) { return false; }
            }
        }
        return true;
    }

    /// Visit the values representing the uintersection
    fn intersection(&self, other: &SmallIntSet, f: &fn(&uint) -> bool) -> bool {
        self.each(|v| !other.contains(v) || f(v))
    }

    /// Visit the values representing the union
    fn union(&self, other: &SmallIntSet, f: &fn(&uint) -> bool) -> bool {
        let len = cmp::max(self.map.v.len() ,other.map.v.len());

        for uint::range(0, len) |i| {
            if self.contains(&i) || other.contains(&i) {
                if !f(&i) { return false; }
            }
        }
        return true;
    }
}

impl SmallIntSet {
    /// Create an empty SmallIntSet
    pub fn new() -> SmallIntSet { SmallIntSet{map: SmallIntMap::new()} }
}

#[cfg(test)]
mod tests {
    use core::prelude::*;

    use super::SmallIntMap;

    use core::local_data;
    use core::rand;
    use core::uint;
    use core::vec;

    #[test]
    fn test_find_mut() {
        let mut m = SmallIntMap::new();
        assert!(m.insert(1, 12));
        assert!(m.insert(2, 8));
        assert!(m.insert(5, 14));
        let new = 100;
        match m.find_mut(&5) {
            None => fail!(), Some(x) => *x = new
        }
        assert_eq!(m.find(&5), Some(&new));
    }

    #[test]
    fn test_len() {
        let mut map = SmallIntMap::new();
        assert_eq!(map.len(), 0);
        assert!(map.is_empty());
        assert!(map.insert(5, 20));
        assert_eq!(map.len(), 1);
        assert!(!map.is_empty());
        assert!(map.insert(11, 12));
        assert_eq!(map.len(), 2);
        assert!(!map.is_empty());
        assert!(map.insert(14, 22));
        assert_eq!(map.len(), 3);
        assert!(!map.is_empty());
    }

    #[test]
    fn test_clear() {
        let mut map = SmallIntMap::new();
        assert!(map.insert(5, 20));
        assert!(map.insert(11, 12));
        assert!(map.insert(14, 22));
        map.clear();
        assert!(map.is_empty());
        assert!(map.find(&5).is_none());
        assert!(map.find(&11).is_none());
        assert!(map.find(&14).is_none());
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
        assert_eq!(map.find(&3).get(), &10);
        assert_eq!(map.find(&5).get(), &3);
        assert_eq!(map.find(&9).get(), &1);

        // sadly, no sevens were counted
        assert!(map.find(&7).is_none());
    }

    #[test]
    fn test_swap() {
        let mut m = SmallIntMap::new();
        assert_eq!(m.swap(1, 2), None);
        assert_eq!(m.swap(1, 3), Some(2));
        assert_eq!(m.swap(1, 4), Some(3));
    }

    #[test]
    fn test_pop() {
        let mut m = SmallIntMap::new();
        m.insert(1, 2);
        assert_eq!(m.pop(&1), Some(2));
        assert_eq!(m.pop(&1), None);
    }
}

#[cfg(test)]
mod test_set {
    use core::prelude::*;

    use super::SmallIntSet;

    use core::vec;

    #[test]
    fn test_disjoint() {
        let mut xs = SmallIntSet::new();
        let mut ys = SmallIntSet::new();
        assert!(xs.is_disjoint(&ys));
        assert!(ys.is_disjoint(&xs));
        assert!(xs.insert(5));
        assert!(ys.insert(11));
        assert!(xs.is_disjoint(&ys));
        assert!(ys.is_disjoint(&xs));
        assert!(xs.insert(7));
        assert!(xs.insert(19));
        assert!(xs.insert(4));
        assert!(ys.insert(2));
        assert!(xs.is_disjoint(&ys));
        assert!(ys.is_disjoint(&xs));
        assert!(ys.insert(7));
        assert!(!xs.is_disjoint(&ys));
        assert!(!ys.is_disjoint(&xs));
    }

    #[test]
    fn test_subset_and_superset() {
        let mut a = SmallIntSet::new();
        assert!(a.insert(0));
        assert!(a.insert(5));
        assert!(a.insert(11));
        assert!(a.insert(7));

        let mut b = SmallIntSet::new();
        assert!(b.insert(0));
        assert!(b.insert(7));
        assert!(b.insert(19));
        assert!(b.insert(250));
        assert!(b.insert(11));
        assert!(b.insert(200));

        assert!(!a.is_subset(&b));
        assert!(!a.is_superset(&b));
        assert!(!b.is_subset(&a));
        assert!(!b.is_superset(&a));

        assert!(b.insert(5));

        assert!(a.is_subset(&b));
        assert!(!a.is_superset(&b));
        assert!(!b.is_subset(&a));
        assert!(b.is_superset(&a));
    }

    #[test]
    fn test_intersection() {
        let mut a = SmallIntSet::new();
        let mut b = SmallIntSet::new();

        assert!(a.insert(11));
        assert!(a.insert(1));
        assert!(a.insert(3));
        assert!(a.insert(77));
        assert!(a.insert(103));
        assert!(a.insert(5));

        assert!(b.insert(2));
        assert!(b.insert(11));
        assert!(b.insert(77));
        assert!(b.insert(5));
        assert!(b.insert(3));

        let mut i = 0;
        let expected = [3, 5, 11, 77];
        for a.intersection(&b) |x| {
            assert!(vec::contains(expected, x));
            i += 1
        }
        assert_eq!(i, expected.len());
    }

    #[test]
    fn test_difference() {
        let mut a = SmallIntSet::new();
        let mut b = SmallIntSet::new();

        assert!(a.insert(1));
        assert!(a.insert(3));
        assert!(a.insert(5));
        assert!(a.insert(9));
        assert!(a.insert(11));

        assert!(b.insert(3));
        assert!(b.insert(9));

        let mut i = 0;
        let expected = [1, 5, 11];
        for a.difference(&b) |x| {
            assert!(vec::contains(expected, x));
            i += 1
        }
        assert_eq!(i, expected.len());
    }

    #[test]
    fn test_symmetric_difference() {
        let mut a = SmallIntSet::new();
        let mut b = SmallIntSet::new();

        assert!(a.insert(1));
        assert!(a.insert(3));
        assert!(a.insert(5));
        assert!(a.insert(9));
        assert!(a.insert(11));

        assert!(b.insert(3));
        assert!(b.insert(9));
        assert!(b.insert(14));
        assert!(b.insert(22));

        let mut i = 0;
        let expected = [1, 5, 11, 14, 22];
        for a.symmetric_difference(&b) |x| {
            assert!(vec::contains(expected, x));
            i += 1
        }
        assert_eq!(i, expected.len());
    }

    #[test]
    fn test_union() {
        let mut a = SmallIntSet::new();
        let mut b = SmallIntSet::new();

        assert!(a.insert(1));
        assert!(a.insert(3));
        assert!(a.insert(5));
        assert!(a.insert(9));
        assert!(a.insert(11));
        assert!(a.insert(16));
        assert!(a.insert(19));
        assert!(a.insert(24));

        assert!(b.insert(1));
        assert!(b.insert(5));
        assert!(b.insert(9));
        assert!(b.insert(13));
        assert!(b.insert(19));

        let mut i = 0;
        let expected = [1, 3, 5, 9, 11, 13, 16, 19, 24];
        for a.union(&b) |x| {
            assert!(vec::contains(expected, x));
            i += 1
        }
        assert_eq!(i, expected.len());
    }
}
