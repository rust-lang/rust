// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
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

#![allow(missing_doc)]

use core::prelude::*;

use core::default::Default;
use core::fmt;
use core::iter::{Enumerate, FilterMap};
use core::mem::replace;

use {Collection, Mutable, Map, MutableMap, MutableSeq};
use {vec, slice};
use vec::Vec;

#[allow(missing_doc)]
pub struct SmallIntMap<T> {
    v: Vec<Option<T>>,
}

impl<V> Collection for SmallIntMap<V> {
    /// Return the number of elements in the map
    fn len(&self) -> uint {
        self.v.iter().filter(|elt| elt.is_some()).count()
    }

    /// Return true if there are no elements in the map
    fn is_empty(&self) -> bool {
        self.v.iter().all(|elt| elt.is_none())
    }
}

impl<V> Mutable for SmallIntMap<V> {
    /// Clear the map, removing all key-value pairs.
    fn clear(&mut self) { self.v.clear() }
}

impl<V> Map<uint, V> for SmallIntMap<V> {
    /// Return a reference to the value corresponding to the key
    fn find<'a>(&'a self, key: &uint) -> Option<&'a V> {
        if *key < self.v.len() {
            match *self.v.get(*key) {
              Some(ref value) => Some(value),
              None => None
            }
        } else {
            None
        }
    }
}

impl<V> MutableMap<uint, V> for SmallIntMap<V> {
    /// Return a mutable reference to the value corresponding to the key
    fn find_mut<'a>(&'a mut self, key: &uint) -> Option<&'a mut V> {
        if *key < self.v.len() {
            match *self.v.get_mut(*key) {
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
            self.v.grow_fn(key - len + 1, |_| None);
        }
        *self.v.get_mut(key) = Some(value);
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
        self.v.get_mut(*key).take()
    }
}

impl<V> Default for SmallIntMap<V> {
    #[inline]
    fn default() -> SmallIntMap<V> { SmallIntMap::new() }
}

impl<V> SmallIntMap<V> {
    /// Create an empty SmallIntMap
    pub fn new() -> SmallIntMap<V> { SmallIntMap{v: vec!()} }

    /// Create an empty SmallIntMap with capacity `capacity`
    pub fn with_capacity(capacity: uint) -> SmallIntMap<V> {
        SmallIntMap { v: Vec::with_capacity(capacity) }
    }

    pub fn get<'a>(&'a self, key: &uint) -> &'a V {
        self.find(key).expect("key not present")
    }

    /// An iterator visiting all key-value pairs in ascending order by the keys.
    /// Iterator element type is (uint, &'r V)
    pub fn iter<'r>(&'r self) -> Entries<'r, V> {
        Entries {
            front: 0,
            back: self.v.len(),
            iter: self.v.iter()
        }
    }

    /// An iterator visiting all key-value pairs in ascending order by the keys,
    /// with mutable references to the values
    /// Iterator element type is (uint, &'r mut V)
    pub fn mut_iter<'r>(&'r mut self) -> MutEntries<'r, V> {
        MutEntries {
            front: 0,
            back: self.v.len(),
            iter: self.v.mut_iter()
        }
    }

    /// Empties the hash map, moving all values into the specified closure
    pub fn move_iter(&mut self)
        -> FilterMap<(uint, Option<V>), (uint, V),
                Enumerate<vec::MoveItems<Option<V>>>>
    {
        let values = replace(&mut self.v, vec!());
        values.move_iter().enumerate().filter_map(|(i, v)| {
            v.map(|v| (i, v))
        })
    }
}

impl<V:Clone> SmallIntMap<V> {
    pub fn update_with_key(&mut self,
                           key: uint,
                           val: V,
                           ff: |uint, V, V| -> V)
                           -> bool {
        let new_val = match self.find(&key) {
            None => val,
            Some(orig) => ff(key, (*orig).clone(), val)
        };
        self.insert(key, new_val)
    }

    pub fn update(&mut self, key: uint, newval: V, ff: |V, V| -> V) -> bool {
        self.update_with_key(key, newval, |_k, v, v1| ff(v,v1))
    }
}

impl<V: fmt::Show> fmt::Show for SmallIntMap<V> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        try!(write!(f, "{{"));

        for (i, (k, v)) in self.iter().enumerate() {
            if i != 0 { try!(write!(f, ", ")); }
            try!(write!(f, "{}: {}", k, *v));
        }

        write!(f, "}}")
    }
}

macro_rules! iterator {
    (impl $name:ident -> $elem:ty, $getter:ident) => {
        impl<'a, T> Iterator<$elem> for $name<'a, T> {
            #[inline]
            fn next(&mut self) -> Option<$elem> {
                while self.front < self.back {
                    match self.iter.next() {
                        Some(elem) => {
                            if elem.is_some() {
                                let index = self.front;
                                self.front += 1;
                                return Some((index, elem. $getter ()));
                            }
                        }
                        _ => ()
                    }
                    self.front += 1;
                }
                None
            }

            #[inline]
            fn size_hint(&self) -> (uint, Option<uint>) {
                (0, Some(self.back - self.front))
            }
        }
    }
}

macro_rules! double_ended_iterator {
    (impl $name:ident -> $elem:ty, $getter:ident) => {
        impl<'a, T> DoubleEndedIterator<$elem> for $name<'a, T> {
            #[inline]
            fn next_back(&mut self) -> Option<$elem> {
                while self.front < self.back {
                    match self.iter.next_back() {
                        Some(elem) => {
                            if elem.is_some() {
                                self.back -= 1;
                                return Some((self.back, elem. $getter ()));
                            }
                        }
                        _ => ()
                    }
                    self.back -= 1;
                }
                None
            }
        }
    }
}

pub struct Entries<'a, T> {
    front: uint,
    back: uint,
    iter: slice::Items<'a, Option<T>>
}

iterator!(impl Entries -> (uint, &'a T), get_ref)
double_ended_iterator!(impl Entries -> (uint, &'a T), get_ref)

pub struct MutEntries<'a, T> {
    front: uint,
    back: uint,
    iter: slice::MutItems<'a, Option<T>>
}

iterator!(impl MutEntries -> (uint, &'a mut T), get_mut_ref)
double_ended_iterator!(impl MutEntries -> (uint, &'a mut T), get_mut_ref)

#[cfg(test)]
mod test_map {
    use std::prelude::*;

    use {Map, MutableMap, Mutable};
    use super::SmallIntMap;

    #[test]
    fn test_find_mut() {
        let mut m = SmallIntMap::new();
        assert!(m.insert(1, 12i));
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
        assert!(map.insert(5, 20i));
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
        assert!(map.insert(5, 20i));
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

        // given a new key, initialize it with this new count,
        // given an existing key, add more to its count
        fn add_more_to_count(_k: uint, v0: uint, v1: uint) -> uint {
            v0 + v1
        }

        fn add_more_to_count_simple(v0: uint, v1: uint) -> uint {
            v0 + v1
        }

        // count integers
        map.update(3, 1, add_more_to_count_simple);
        map.update_with_key(9, 1, add_more_to_count);
        map.update(3, 7, add_more_to_count_simple);
        map.update_with_key(5, 3, add_more_to_count);
        map.update_with_key(3, 2, add_more_to_count);

        // check the total counts
        assert_eq!(map.find(&3).unwrap(), &10);
        assert_eq!(map.find(&5).unwrap(), &3);
        assert_eq!(map.find(&9).unwrap(), &1);

        // sadly, no sevens were counted
        assert!(map.find(&7).is_none());
    }

    #[test]
    fn test_swap() {
        let mut m = SmallIntMap::new();
        assert_eq!(m.swap(1, 2i), None);
        assert_eq!(m.swap(1, 3i), Some(2));
        assert_eq!(m.swap(1, 4i), Some(3));
    }

    #[test]
    fn test_pop() {
        let mut m = SmallIntMap::new();
        m.insert(1, 2i);
        assert_eq!(m.pop(&1), Some(2));
        assert_eq!(m.pop(&1), None);
    }

    #[test]
    fn test_iterator() {
        let mut m = SmallIntMap::new();

        assert!(m.insert(0, 1i));
        assert!(m.insert(1, 2));
        assert!(m.insert(3, 5));
        assert!(m.insert(6, 10));
        assert!(m.insert(10, 11));

        let mut it = m.iter();
        assert_eq!(it.size_hint(), (0, Some(11)));
        assert_eq!(it.next().unwrap(), (0, &1));
        assert_eq!(it.size_hint(), (0, Some(10)));
        assert_eq!(it.next().unwrap(), (1, &2));
        assert_eq!(it.size_hint(), (0, Some(9)));
        assert_eq!(it.next().unwrap(), (3, &5));
        assert_eq!(it.size_hint(), (0, Some(7)));
        assert_eq!(it.next().unwrap(), (6, &10));
        assert_eq!(it.size_hint(), (0, Some(4)));
        assert_eq!(it.next().unwrap(), (10, &11));
        assert_eq!(it.size_hint(), (0, Some(0)));
        assert!(it.next().is_none());
    }

    #[test]
    fn test_iterator_size_hints() {
        let mut m = SmallIntMap::new();

        assert!(m.insert(0, 1i));
        assert!(m.insert(1, 2));
        assert!(m.insert(3, 5));
        assert!(m.insert(6, 10));
        assert!(m.insert(10, 11));

        assert_eq!(m.iter().size_hint(), (0, Some(11)));
        assert_eq!(m.iter().rev().size_hint(), (0, Some(11)));
        assert_eq!(m.mut_iter().size_hint(), (0, Some(11)));
        assert_eq!(m.mut_iter().rev().size_hint(), (0, Some(11)));
    }

    #[test]
    fn test_mut_iterator() {
        let mut m = SmallIntMap::new();

        assert!(m.insert(0, 1i));
        assert!(m.insert(1, 2));
        assert!(m.insert(3, 5));
        assert!(m.insert(6, 10));
        assert!(m.insert(10, 11));

        for (k, v) in m.mut_iter() {
            *v += k as int;
        }

        let mut it = m.iter();
        assert_eq!(it.next().unwrap(), (0, &1));
        assert_eq!(it.next().unwrap(), (1, &3));
        assert_eq!(it.next().unwrap(), (3, &8));
        assert_eq!(it.next().unwrap(), (6, &16));
        assert_eq!(it.next().unwrap(), (10, &21));
        assert!(it.next().is_none());
    }

    #[test]
    fn test_rev_iterator() {
        let mut m = SmallIntMap::new();

        assert!(m.insert(0, 1i));
        assert!(m.insert(1, 2));
        assert!(m.insert(3, 5));
        assert!(m.insert(6, 10));
        assert!(m.insert(10, 11));

        let mut it = m.iter().rev();
        assert_eq!(it.next().unwrap(), (10, &11));
        assert_eq!(it.next().unwrap(), (6, &10));
        assert_eq!(it.next().unwrap(), (3, &5));
        assert_eq!(it.next().unwrap(), (1, &2));
        assert_eq!(it.next().unwrap(), (0, &1));
        assert!(it.next().is_none());
    }

    #[test]
    fn test_mut_rev_iterator() {
        let mut m = SmallIntMap::new();

        assert!(m.insert(0, 1i));
        assert!(m.insert(1, 2));
        assert!(m.insert(3, 5));
        assert!(m.insert(6, 10));
        assert!(m.insert(10, 11));

        for (k, v) in m.mut_iter().rev() {
            *v += k as int;
        }

        let mut it = m.iter();
        assert_eq!(it.next().unwrap(), (0, &1));
        assert_eq!(it.next().unwrap(), (1, &3));
        assert_eq!(it.next().unwrap(), (3, &8));
        assert_eq!(it.next().unwrap(), (6, &16));
        assert_eq!(it.next().unwrap(), (10, &21));
        assert!(it.next().is_none());
    }

    #[test]
    fn test_move_iter() {
        let mut m = SmallIntMap::new();
        m.insert(1, box 2i);
        let mut called = false;
        for (k, v) in m.move_iter() {
            assert!(!called);
            called = true;
            assert_eq!(k, 1);
            assert_eq!(v, box 2i);
        }
        assert!(called);
        m.insert(2, box 1i);
    }

    #[test]
    fn test_show() {
        let mut map = SmallIntMap::new();
        let empty = SmallIntMap::<int>::new();

        map.insert(1, 2i);
        map.insert(3, 4i);

        let map_str = map.to_string();
        let map_str = map_str.as_slice();
        assert!(map_str == "{1: 2, 3: 4}" || map_str == "{3: 4, 1: 2}");
        assert_eq!(format!("{}", empty), "{}".to_string());
    }
}

#[cfg(test)]
mod bench {
    extern crate test;
    use self::test::Bencher;
    use super::SmallIntMap;
    use deque::bench::{insert_rand_n, insert_seq_n, find_rand_n, find_seq_n};

    // Find seq
    #[bench]
    pub fn insert_rand_100(b: &mut Bencher) {
        let mut m : SmallIntMap<uint> = SmallIntMap::new();
        insert_rand_n(100, &mut m, b);
    }

    #[bench]
    pub fn insert_rand_10_000(b: &mut Bencher) {
        let mut m : SmallIntMap<uint> = SmallIntMap::new();
        insert_rand_n(10_000, &mut m, b);
    }

    // Insert seq
    #[bench]
    pub fn insert_seq_100(b: &mut Bencher) {
        let mut m : SmallIntMap<uint> = SmallIntMap::new();
        insert_seq_n(100, &mut m, b);
    }

    #[bench]
    pub fn insert_seq_10_000(b: &mut Bencher) {
        let mut m : SmallIntMap<uint> = SmallIntMap::new();
        insert_seq_n(10_000, &mut m, b);
    }

    // Find rand
    #[bench]
    pub fn find_rand_100(b: &mut Bencher) {
        let mut m : SmallIntMap<uint> = SmallIntMap::new();
        find_rand_n(100, &mut m, b);
    }

    #[bench]
    pub fn find_rand_10_000(b: &mut Bencher) {
        let mut m : SmallIntMap<uint> = SmallIntMap::new();
        find_rand_n(10_000, &mut m, b);
    }

    // Find seq
    #[bench]
    pub fn find_seq_100(b: &mut Bencher) {
        let mut m : SmallIntMap<uint> = SmallIntMap::new();
        find_seq_n(100, &mut m, b);
    }

    #[bench]
    pub fn find_seq_10_000(b: &mut Bencher) {
        let mut m : SmallIntMap<uint> = SmallIntMap::new();
        find_seq_n(10_000, &mut m, b);
    }
}
