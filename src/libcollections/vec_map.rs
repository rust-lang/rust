// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! A simple map based on a vector for small integer keys. Space requirements
//! are O(highest integer key).

#![allow(missing_docs)]

use core::prelude::*;

use core::default::Default;
use core::fmt;
use core::hash::{Hash, Writer};
use core::iter;
use core::iter::{Enumerate, FilterMap, Map};
use core::mem::replace;

use {vec, slice};
use vec::Vec;

// FIXME(conventions): capacity management???

/// A map optimized for small integer keys.
///
/// # Examples
///
/// ```
/// use std::collections::VecMap;
///
/// let mut months = VecMap::new();
/// months.insert(1, "Jan");
/// months.insert(2, "Feb");
/// months.insert(3, "Mar");
///
/// if !months.contains_key(&12) {
///     println!("The end is near!");
/// }
///
/// assert_eq!(months.get(&1), Some(&"Jan"));
///
/// match months.get_mut(&3) {
///     Some(value) => *value = "Venus",
///     None => (),
/// }
///
/// assert_eq!(months.get(&3), Some(&"Venus"));
///
/// // Print out all months
/// for (key, value) in months.iter() {
///     println!("month {} is {}", key, value);
/// }
///
/// months.clear();
/// assert!(months.is_empty());
/// ```
#[deriving(PartialEq, Eq)]
pub struct VecMap<V> {
    v: Vec<Option<V>>,
}

#[stable]
impl<V> Default for VecMap<V> {
    #[stable]
    #[inline]
    fn default() -> VecMap<V> { VecMap::new() }
}

impl<V:Clone> Clone for VecMap<V> {
    #[inline]
    fn clone(&self) -> VecMap<V> {
        VecMap { v: self.v.clone() }
    }

    #[inline]
    fn clone_from(&mut self, source: &VecMap<V>) {
        self.v.clone_from(&source.v);
    }
}

impl<S: Writer, V: Hash<S>> Hash<S> for VecMap<V> {
    fn hash(&self, state: &mut S) {
        // In order to not traverse the `VecMap` twice, count the elements
        // during iteration.
        let mut count: uint = 0;
        for elt in self.iter() {
            elt.hash(state);
            count += 1;
        }
        count.hash(state);
    }
}

impl<V> VecMap<V> {
    /// Creates an empty `VecMap`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::VecMap;
    /// let mut map: VecMap<&str> = VecMap::new();
    /// ```
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn new() -> VecMap<V> { VecMap { v: vec![] } }

    /// Creates an empty `VecMap` with space for at least `capacity`
    /// elements before resizing.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::VecMap;
    /// let mut map: VecMap<&str> = VecMap::with_capacity(10);
    /// ```
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn with_capacity(capacity: uint) -> VecMap<V> {
        VecMap { v: Vec::with_capacity(capacity) }
    }

    /// Returns the number of elements the `VecMap` can hold without
    /// reallocating.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::VecMap;
    /// let map: VecMap<String> = VecMap::with_capacity(10);
    /// assert!(map.capacity() >= 10);
    /// ```
    #[inline]
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn capacity(&self) -> uint {
        self.v.capacity()
    }

    /// Returns an iterator visiting all keys in ascending order by the keys.
    /// The iterator's element type is `uint`.
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn keys<'r>(&'r self) -> Keys<'r, V> {
        fn first<A, B>((a, _): (A, B)) -> A { a }
        let first: fn((uint, &'r V)) -> uint = first; // coerce to fn pointer

        Keys { iter: self.iter().map(first) }
    }

    /// Returns an iterator visiting all values in ascending order by the keys.
    /// The iterator's element type is `&'r V`.
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn values<'r>(&'r self) -> Values<'r, V> {
        fn second<A, B>((_, b): (A, B)) -> B { b }
        let second: fn((uint, &'r V)) -> &'r V = second; // coerce to fn pointer

        Values { iter: self.iter().map(second) }
    }

    /// Returns an iterator visiting all key-value pairs in ascending order by the keys.
    /// The iterator's element type is `(uint, &'r V)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::VecMap;
    ///
    /// let mut map = VecMap::new();
    /// map.insert(1, "a");
    /// map.insert(3, "c");
    /// map.insert(2, "b");
    ///
    /// // Print `1: a` then `2: b` then `3: c`
    /// for (key, value) in map.iter() {
    ///     println!("{}: {}", key, value);
    /// }
    /// ```
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn iter<'r>(&'r self) -> Iter<'r, V> {
        Iter {
            front: 0,
            back: self.v.len(),
            iter: self.v.iter()
        }
    }

    /// Returns an iterator visiting all key-value pairs in ascending order by the keys,
    /// with mutable references to the values.
    /// The iterator's element type is `(uint, &'r mut V)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::VecMap;
    ///
    /// let mut map = VecMap::new();
    /// map.insert(1, "a");
    /// map.insert(2, "b");
    /// map.insert(3, "c");
    ///
    /// for (key, value) in map.iter_mut() {
    ///     *value = "x";
    /// }
    ///
    /// for (key, value) in map.iter() {
    ///     assert_eq!(value, &"x");
    /// }
    /// ```
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn iter_mut<'r>(&'r mut self) -> IterMut<'r, V> {
        IterMut {
            front: 0,
            back: self.v.len(),
            iter: self.v.iter_mut()
        }
    }

    /// Returns an iterator visiting all key-value pairs in ascending order by
    /// the keys, emptying (but not consuming) the original `VecMap`.
    /// The iterator's element type is `(uint, &'r V)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::VecMap;
    ///
    /// let mut map = VecMap::new();
    /// map.insert(1, "a");
    /// map.insert(3, "c");
    /// map.insert(2, "b");
    ///
    /// // Not possible with .iter()
    /// let vec: Vec<(uint, &str)> = map.into_iter().collect();
    ///
    /// assert_eq!(vec, vec![(1, "a"), (2, "b"), (3, "c")]);
    /// ```
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn into_iter(&mut self) -> IntoIter<V> {
        fn filter<A>((i, v): (uint, Option<A>)) -> Option<(uint, A)> {
            v.map(|v| (i, v))
        }
        let filter: fn((uint, Option<V>)) -> Option<(uint, V)> = filter; // coerce to fn ptr

        let values = replace(&mut self.v, vec!());
        IntoIter { iter: values.into_iter().enumerate().filter_map(filter) }
    }

    /// Return the number of elements in the map.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::VecMap;
    ///
    /// let mut a = VecMap::new();
    /// assert_eq!(a.len(), 0);
    /// a.insert(1, "a");
    /// assert_eq!(a.len(), 1);
    /// ```
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn len(&self) -> uint {
        self.v.iter().filter(|elt| elt.is_some()).count()
    }

    /// Return true if the map contains no elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::VecMap;
    ///
    /// let mut a = VecMap::new();
    /// assert!(a.is_empty());
    /// a.insert(1, "a");
    /// assert!(!a.is_empty());
    /// ```
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn is_empty(&self) -> bool {
        self.v.iter().all(|elt| elt.is_none())
    }

    /// Clears the map, removing all key-value pairs.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::VecMap;
    ///
    /// let mut a = VecMap::new();
    /// a.insert(1, "a");
    /// a.clear();
    /// assert!(a.is_empty());
    /// ```
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn clear(&mut self) { self.v.clear() }

    /// Deprecated: Renamed to `get`.
    #[deprecated = "Renamed to `get`"]
    pub fn find(&self, key: &uint) -> Option<&V> {
        self.get(key)
    }

    /// Returns a reference to the value corresponding to the key.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::VecMap;
    ///
    /// let mut map = VecMap::new();
    /// map.insert(1, "a");
    /// assert_eq!(map.get(&1), Some(&"a"));
    /// assert_eq!(map.get(&2), None);
    /// ```
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn get(&self, key: &uint) -> Option<&V> {
        if *key < self.v.len() {
            match self.v[*key] {
              Some(ref value) => Some(value),
              None => None
            }
        } else {
            None
        }
    }

    /// Returns true if the map contains a value for the specified key.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::VecMap;
    ///
    /// let mut map = VecMap::new();
    /// map.insert(1, "a");
    /// assert_eq!(map.contains_key(&1), true);
    /// assert_eq!(map.contains_key(&2), false);
    /// ```
    #[inline]
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn contains_key(&self, key: &uint) -> bool {
        self.get(key).is_some()
    }

    /// Deprecated: Renamed to `get_mut`.
    #[deprecated = "Renamed to `get_mut`"]
    pub fn find_mut(&mut self, key: &uint) -> Option<&mut V> {
        self.get_mut(key)
    }

    /// Returns a mutable reference to the value corresponding to the key.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::VecMap;
    ///
    /// let mut map = VecMap::new();
    /// map.insert(1, "a");
    /// match map.get_mut(&1) {
    ///     Some(x) => *x = "b",
    ///     None => (),
    /// }
    /// assert_eq!(map[1], "b");
    /// ```
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn get_mut(&mut self, key: &uint) -> Option<&mut V> {
        if *key < self.v.len() {
            match *(&mut self.v[*key]) {
              Some(ref mut value) => Some(value),
              None => None
            }
        } else {
            None
        }
    }

    /// Deprecated: Renamed to `insert`.
    #[deprecated = "Renamed to `insert`"]
    pub fn swap(&mut self, key: uint, value: V) -> Option<V> {
        self.insert(key, value)
    }

    /// Inserts a key-value pair from the map. If the key already had a value
    /// present in the map, that value is returned. Otherwise, `None` is returned.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::VecMap;
    ///
    /// let mut map = VecMap::new();
    /// assert_eq!(map.insert(37, "a"), None);
    /// assert_eq!(map.is_empty(), false);
    ///
    /// map.insert(37, "b");
    /// assert_eq!(map.insert(37, "c"), Some("b"));
    /// assert_eq!(map[37], "c");
    /// ```
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn insert(&mut self, key: uint, value: V) -> Option<V> {
        let len = self.v.len();
        if len <= key {
            self.v.grow_fn(key - len + 1, |_| None);
        }
        replace(&mut self.v[key], Some(value))
    }

    /// Deprecated: Renamed to `remove`.
    #[deprecated = "Renamed to `remove`"]
    pub fn pop(&mut self, key: &uint) -> Option<V> {
        self.remove(key)
    }

    /// Removes a key from the map, returning the value at the key if the key
    /// was previously in the map.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::VecMap;
    ///
    /// let mut map = VecMap::new();
    /// map.insert(1, "a");
    /// assert_eq!(map.remove(&1), Some("a"));
    /// assert_eq!(map.remove(&1), None);
    /// ```
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn remove(&mut self, key: &uint) -> Option<V> {
        if *key >= self.v.len() {
            return None;
        }
        self.v[*key].take()
    }
}

impl<V:Clone> VecMap<V> {
    /// Updates a value in the map. If the key already exists in the map,
    /// modifies the value with `ff` taking `oldval, newval`.
    /// Otherwise, sets the value to `newval`.
    /// Returns `true` if the key did not already exist in the map.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::VecMap;
    ///
    /// let mut map = VecMap::new();
    ///
    /// // Key does not exist, will do a simple insert
    /// assert!(map.update(1, vec![1i, 2], |mut old, new| { old.extend(new.into_iter()); old }));
    /// assert_eq!(map[1], vec![1i, 2]);
    ///
    /// // Key exists, update the value
    /// assert!(!map.update(1, vec![3i, 4], |mut old, new| { old.extend(new.into_iter()); old }));
    /// assert_eq!(map[1], vec![1i, 2, 3, 4]);
    /// ```
    pub fn update<F>(&mut self, key: uint, newval: V, ff: F) -> bool where F: FnOnce(V, V) -> V {
        self.update_with_key(key, newval, move |_k, v, v1| ff(v,v1))
    }

    /// Updates a value in the map. If the key already exists in the map,
    /// modifies the value with `ff` taking `key, oldval, newval`.
    /// Otherwise, sets the value to `newval`.
    /// Returns `true` if the key did not already exist in the map.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::VecMap;
    ///
    /// let mut map = VecMap::new();
    ///
    /// // Key does not exist, will do a simple insert
    /// assert!(map.update_with_key(7, 10, |key, old, new| (old + new) % key));
    /// assert_eq!(map[7], 10);
    ///
    /// // Key exists, update the value
    /// assert!(!map.update_with_key(7, 20, |key, old, new| (old + new) % key));
    /// assert_eq!(map[7], 2);
    /// ```
    pub fn update_with_key<F>(&mut self, key: uint, val: V, ff: F) -> bool where
        F: FnOnce(uint, V, V) -> V
    {
        let new_val = match self.get(&key) {
            None => val,
            Some(orig) => ff(key, (*orig).clone(), val)
        };
        self.insert(key, new_val).is_none()
    }
}

impl<V: PartialOrd> PartialOrd for VecMap<V> {
    #[inline]
    fn partial_cmp(&self, other: &VecMap<V>) -> Option<Ordering> {
        iter::order::partial_cmp(self.iter(), other.iter())
    }
}

impl<V: Ord> Ord for VecMap<V> {
    #[inline]
    fn cmp(&self, other: &VecMap<V>) -> Ordering {
        iter::order::cmp(self.iter(), other.iter())
    }
}

impl<V: fmt::Show> fmt::Show for VecMap<V> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        try!(write!(f, "{{"));

        for (i, (k, v)) in self.iter().enumerate() {
            if i != 0 { try!(write!(f, ", ")); }
            try!(write!(f, "{}: {}", k, *v));
        }

        write!(f, "}}")
    }
}

impl<V> FromIterator<(uint, V)> for VecMap<V> {
    fn from_iter<Iter: Iterator<(uint, V)>>(iter: Iter) -> VecMap<V> {
        let mut map = VecMap::new();
        map.extend(iter);
        map
    }
}

impl<V> Extend<(uint, V)> for VecMap<V> {
    fn extend<Iter: Iterator<(uint, V)>>(&mut self, mut iter: Iter) {
        for (k, v) in iter {
            self.insert(k, v);
        }
    }
}

impl<V> Index<uint, V> for VecMap<V> {
    #[inline]
    fn index<'a>(&'a self, i: &uint) -> &'a V {
        self.get(i).expect("key not present")
    }
}

impl<V> IndexMut<uint, V> for VecMap<V> {
    #[inline]
    fn index_mut<'a>(&'a mut self, i: &uint) -> &'a mut V {
        self.get_mut(i).expect("key not present")
    }
}

macro_rules! iterator {
    (impl $name:ident -> $elem:ty, $($getter:ident),+) => {
        impl<'a, V> Iterator<$elem> for $name<'a, V> {
            #[inline]
            fn next(&mut self) -> Option<$elem> {
                while self.front < self.back {
                    match self.iter.next() {
                        Some(elem) => {
                            match elem$(. $getter ())+ {
                                Some(x) => {
                                    let index = self.front;
                                    self.front += 1;
                                    return Some((index, x));
                                },
                                None => {},
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
    (impl $name:ident -> $elem:ty, $($getter:ident),+) => {
        impl<'a, V> DoubleEndedIterator<$elem> for $name<'a, V> {
            #[inline]
            fn next_back(&mut self) -> Option<$elem> {
                while self.front < self.back {
                    match self.iter.next_back() {
                        Some(elem) => {
                            match elem$(. $getter ())+ {
                                Some(x) => {
                                    self.back -= 1;
                                    return Some((self.back, x));
                                },
                                None => {},
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

/// An iterator over the key-value pairs of a map.
pub struct Iter<'a, V:'a> {
    front: uint,
    back: uint,
    iter: slice::Iter<'a, Option<V>>
}

iterator! { impl Iter -> (uint, &'a V), as_ref }
double_ended_iterator! { impl Iter -> (uint, &'a V), as_ref }

/// An iterator over the key-value pairs of a map, with the
/// values being mutable.
pub struct IterMut<'a, V:'a> {
    front: uint,
    back: uint,
    iter: slice::IterMut<'a, Option<V>>
}

iterator! { impl IterMut -> (uint, &'a mut V), as_mut }
double_ended_iterator! { impl IterMut -> (uint, &'a mut V), as_mut }

/// An iterator over the keys of a map.
pub struct Keys<'a, V: 'a> {
    iter: Map<(uint, &'a V), uint, Iter<'a, V>, fn((uint, &'a V)) -> uint>
}

/// An iterator over the values of a map.
pub struct Values<'a, V: 'a> {
    iter: Map<(uint, &'a V), &'a V, Iter<'a, V>, fn((uint, &'a V)) -> &'a V>
}

/// A consuming iterator over the key-value pairs of a map.
pub struct IntoIter<V> {
    iter: FilterMap<
    (uint, Option<V>),
    (uint, V),
    Enumerate<vec::IntoIter<Option<V>>>,
    fn((uint, Option<V>)) -> Option<(uint, V)>>
}

impl<'a, V> Iterator<uint> for Keys<'a, V> {
    fn next(&mut self) -> Option<uint> { self.iter.next() }
    fn size_hint(&self) -> (uint, Option<uint>) { self.iter.size_hint() }
}
impl<'a, V> DoubleEndedIterator<uint> for Keys<'a, V> {
    fn next_back(&mut self) -> Option<uint> { self.iter.next_back() }
}


impl<'a, V> Iterator<&'a V> for Values<'a, V> {
    fn next(&mut self) -> Option<(&'a V)> { self.iter.next() }
    fn size_hint(&self) -> (uint, Option<uint>) { self.iter.size_hint() }
}
impl<'a, V> DoubleEndedIterator<&'a V> for Values<'a, V> {
    fn next_back(&mut self) -> Option<(&'a V)> { self.iter.next_back() }
}


impl<V> Iterator<(uint, V)> for IntoIter<V> {
    fn next(&mut self) -> Option<(uint, V)> { self.iter.next() }
    fn size_hint(&self) -> (uint, Option<uint>) { self.iter.size_hint() }
}
impl<V> DoubleEndedIterator<(uint, V)> for IntoIter<V> {
    fn next_back(&mut self) -> Option<(uint, V)> { self.iter.next_back() }
}

#[cfg(test)]
mod test_map {
    use prelude::*;
    use core::hash::hash;

    use super::VecMap;

    #[test]
    fn test_get_mut() {
        let mut m = VecMap::new();
        assert!(m.insert(1, 12i).is_none());
        assert!(m.insert(2, 8).is_none());
        assert!(m.insert(5, 14).is_none());
        let new = 100;
        match m.get_mut(&5) {
            None => panic!(), Some(x) => *x = new
        }
        assert_eq!(m.get(&5), Some(&new));
    }

    #[test]
    fn test_len() {
        let mut map = VecMap::new();
        assert_eq!(map.len(), 0);
        assert!(map.is_empty());
        assert!(map.insert(5, 20i).is_none());
        assert_eq!(map.len(), 1);
        assert!(!map.is_empty());
        assert!(map.insert(11, 12).is_none());
        assert_eq!(map.len(), 2);
        assert!(!map.is_empty());
        assert!(map.insert(14, 22).is_none());
        assert_eq!(map.len(), 3);
        assert!(!map.is_empty());
    }

    #[test]
    fn test_clear() {
        let mut map = VecMap::new();
        assert!(map.insert(5, 20i).is_none());
        assert!(map.insert(11, 12).is_none());
        assert!(map.insert(14, 22).is_none());
        map.clear();
        assert!(map.is_empty());
        assert!(map.get(&5).is_none());
        assert!(map.get(&11).is_none());
        assert!(map.get(&14).is_none());
    }

    #[test]
    fn test_insert_with_key() {
        let mut map = VecMap::new();

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
        assert_eq!(map.get(&3).unwrap(), &10);
        assert_eq!(map.get(&5).unwrap(), &3);
        assert_eq!(map.get(&9).unwrap(), &1);

        // sadly, no sevens were counted
        assert!(map.get(&7).is_none());
    }

    #[test]
    fn test_insert() {
        let mut m = VecMap::new();
        assert_eq!(m.insert(1, 2i), None);
        assert_eq!(m.insert(1, 3i), Some(2));
        assert_eq!(m.insert(1, 4i), Some(3));
    }

    #[test]
    fn test_remove() {
        let mut m = VecMap::new();
        m.insert(1, 2i);
        assert_eq!(m.remove(&1), Some(2));
        assert_eq!(m.remove(&1), None);
    }

    #[test]
    fn test_keys() {
        let mut map = VecMap::new();
        map.insert(1, 'a');
        map.insert(2, 'b');
        map.insert(3, 'c');
        let keys = map.keys().collect::<Vec<uint>>();
        assert_eq!(keys.len(), 3);
        assert!(keys.contains(&1));
        assert!(keys.contains(&2));
        assert!(keys.contains(&3));
    }

    #[test]
    fn test_values() {
        let mut map = VecMap::new();
        map.insert(1, 'a');
        map.insert(2, 'b');
        map.insert(3, 'c');
        let values = map.values().map(|&v| v).collect::<Vec<char>>();
        assert_eq!(values.len(), 3);
        assert!(values.contains(&'a'));
        assert!(values.contains(&'b'));
        assert!(values.contains(&'c'));
    }

    #[test]
    fn test_iterator() {
        let mut m = VecMap::new();

        assert!(m.insert(0, 1i).is_none());
        assert!(m.insert(1, 2).is_none());
        assert!(m.insert(3, 5).is_none());
        assert!(m.insert(6, 10).is_none());
        assert!(m.insert(10, 11).is_none());

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
        let mut m = VecMap::new();

        assert!(m.insert(0, 1i).is_none());
        assert!(m.insert(1, 2).is_none());
        assert!(m.insert(3, 5).is_none());
        assert!(m.insert(6, 10).is_none());
        assert!(m.insert(10, 11).is_none());

        assert_eq!(m.iter().size_hint(), (0, Some(11)));
        assert_eq!(m.iter().rev().size_hint(), (0, Some(11)));
        assert_eq!(m.iter_mut().size_hint(), (0, Some(11)));
        assert_eq!(m.iter_mut().rev().size_hint(), (0, Some(11)));
    }

    #[test]
    fn test_mut_iterator() {
        let mut m = VecMap::new();

        assert!(m.insert(0, 1i).is_none());
        assert!(m.insert(1, 2).is_none());
        assert!(m.insert(3, 5).is_none());
        assert!(m.insert(6, 10).is_none());
        assert!(m.insert(10, 11).is_none());

        for (k, v) in m.iter_mut() {
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
        let mut m = VecMap::new();

        assert!(m.insert(0, 1i).is_none());
        assert!(m.insert(1, 2).is_none());
        assert!(m.insert(3, 5).is_none());
        assert!(m.insert(6, 10).is_none());
        assert!(m.insert(10, 11).is_none());

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
        let mut m = VecMap::new();

        assert!(m.insert(0, 1i).is_none());
        assert!(m.insert(1, 2).is_none());
        assert!(m.insert(3, 5).is_none());
        assert!(m.insert(6, 10).is_none());
        assert!(m.insert(10, 11).is_none());

        for (k, v) in m.iter_mut().rev() {
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
        let mut m = VecMap::new();
        m.insert(1, box 2i);
        let mut called = false;
        for (k, v) in m.into_iter() {
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
        let mut map = VecMap::new();
        let empty = VecMap::<int>::new();

        map.insert(1, 2i);
        map.insert(3, 4i);

        let map_str = map.to_string();
        assert!(map_str == "{1: 2, 3: 4}" || map_str == "{3: 4, 1: 2}");
        assert_eq!(format!("{}", empty), "{}");
    }

    #[test]
    fn test_clone() {
        let mut a = VecMap::new();

        a.insert(1, 'x');
        a.insert(4, 'y');
        a.insert(6, 'z');

        assert!(a.clone() == a);
    }

    #[test]
    fn test_eq() {
        let mut a = VecMap::new();
        let mut b = VecMap::new();

        assert!(a == b);
        assert!(a.insert(0, 5i).is_none());
        assert!(a != b);
        assert!(b.insert(0, 4i).is_none());
        assert!(a != b);
        assert!(a.insert(5, 19).is_none());
        assert!(a != b);
        assert!(!b.insert(0, 5).is_none());
        assert!(a != b);
        assert!(b.insert(5, 19).is_none());
        assert!(a == b);
    }

    #[test]
    fn test_lt() {
        let mut a = VecMap::new();
        let mut b = VecMap::new();

        assert!(!(a < b) && !(b < a));
        assert!(b.insert(2u, 5i).is_none());
        assert!(a < b);
        assert!(a.insert(2, 7).is_none());
        assert!(!(a < b) && b < a);
        assert!(b.insert(1, 0).is_none());
        assert!(b < a);
        assert!(a.insert(0, 6).is_none());
        assert!(a < b);
        assert!(a.insert(6, 2).is_none());
        assert!(a < b && !(b < a));
    }

    #[test]
    fn test_ord() {
        let mut a = VecMap::new();
        let mut b = VecMap::new();

        assert!(a <= b && a >= b);
        assert!(a.insert(1u, 1i).is_none());
        assert!(a > b && a >= b);
        assert!(b < a && b <= a);
        assert!(b.insert(2, 2).is_none());
        assert!(b > a && b >= a);
        assert!(a < b && a <= b);
    }

    #[test]
    fn test_hash() {
        let mut x = VecMap::new();
        let mut y = VecMap::new();

        assert!(hash(&x) == hash(&y));
        x.insert(1, 'a');
        x.insert(2, 'b');
        x.insert(3, 'c');

        y.insert(3, 'c');
        y.insert(2, 'b');
        y.insert(1, 'a');

        assert!(hash(&x) == hash(&y));

        x.insert(1000, 'd');
        x.remove(&1000);

        assert!(hash(&x) == hash(&y));
    }

    #[test]
    fn test_from_iter() {
        let xs: Vec<(uint, char)> = vec![(1u, 'a'), (2, 'b'), (3, 'c'), (4, 'd'), (5, 'e')];

        let map: VecMap<char> = xs.iter().map(|&x| x).collect();

        for &(k, v) in xs.iter() {
            assert_eq!(map.get(&k), Some(&v));
        }
    }

    #[test]
    fn test_index() {
        let mut map: VecMap<int> = VecMap::new();

        map.insert(1, 2);
        map.insert(2, 1);
        map.insert(3, 4);

        assert_eq!(map[3], 4);
    }

    #[test]
    #[should_fail]
    fn test_index_nonexistent() {
        let mut map: VecMap<int> = VecMap::new();

        map.insert(1, 2);
        map.insert(2, 1);
        map.insert(3, 4);

        map[4];
    }
}

#[cfg(test)]
mod bench {
    use test::Bencher;
    use super::VecMap;
    use bench::{insert_rand_n, insert_seq_n, find_rand_n, find_seq_n};

    #[bench]
    pub fn insert_rand_100(b: &mut Bencher) {
        let mut m : VecMap<uint> = VecMap::new();
        insert_rand_n(100, &mut m, b,
                      |m, i| { m.insert(i, 1); },
                      |m, i| { m.remove(&i); });
    }

    #[bench]
    pub fn insert_rand_10_000(b: &mut Bencher) {
        let mut m : VecMap<uint> = VecMap::new();
        insert_rand_n(10_000, &mut m, b,
                      |m, i| { m.insert(i, 1); },
                      |m, i| { m.remove(&i); });
    }

    // Insert seq
    #[bench]
    pub fn insert_seq_100(b: &mut Bencher) {
        let mut m : VecMap<uint> = VecMap::new();
        insert_seq_n(100, &mut m, b,
                     |m, i| { m.insert(i, 1); },
                     |m, i| { m.remove(&i); });
    }

    #[bench]
    pub fn insert_seq_10_000(b: &mut Bencher) {
        let mut m : VecMap<uint> = VecMap::new();
        insert_seq_n(10_000, &mut m, b,
                     |m, i| { m.insert(i, 1); },
                     |m, i| { m.remove(&i); });
    }

    // Find rand
    #[bench]
    pub fn find_rand_100(b: &mut Bencher) {
        let mut m : VecMap<uint> = VecMap::new();
        find_rand_n(100, &mut m, b,
                    |m, i| { m.insert(i, 1); },
                    |m, i| { m.get(&i); });
    }

    #[bench]
    pub fn find_rand_10_000(b: &mut Bencher) {
        let mut m : VecMap<uint> = VecMap::new();
        find_rand_n(10_000, &mut m, b,
                    |m, i| { m.insert(i, 1); },
                    |m, i| { m.get(&i); });
    }

    // Find seq
    #[bench]
    pub fn find_seq_100(b: &mut Bencher) {
        let mut m : VecMap<uint> = VecMap::new();
        find_seq_n(100, &mut m, b,
                   |m, i| { m.insert(i, 1); },
                   |m, i| { m.get(&i); });
    }

    #[bench]
    pub fn find_seq_10_000(b: &mut Bencher) {
        let mut m : VecMap<uint> = VecMap::new();
        find_seq_n(10_000, &mut m, b,
                   |m, i| { m.insert(i, 1); },
                   |m, i| { m.get(&i); });
    }
}
