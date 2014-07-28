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

#![allow(missing_doc)]

use core::prelude::*;

use core::default::Default;
use core::fmt;
use core::iter;
use core::iter::{Enumerate, FilterMap};
use core::mem::replace;

use {Collection, Mutable, Map, MutableMap, MutableSeq};
use {vec, slice};
use vec::Vec;
use hash;
use hash::Hash;

/// A map optimized for small integer keys.
///
/// # Example
///
/// ```
/// use std::collections::SmallIntMap;
///
/// let mut months = SmallIntMap::new();
/// months.insert(1, "Jan");
/// months.insert(2, "Feb");
/// months.insert(3, "Mar");
///
/// if !months.contains_key(&12) {
///     println!("The end is near!");
/// }
///
/// assert_eq!(months.find(&1), Some(&"Jan"));
///
/// match months.find_mut(&3) {
///     Some(value) => *value = "Venus",
///     None => (),
/// }
///
/// assert_eq!(months.find(&3), Some(&"Venus"));
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
pub struct SmallIntMap<T> {
    v: Vec<Option<T>>,
}

impl<V> Collection for SmallIntMap<V> {
    /// Return the number of elements in the map.
    fn len(&self) -> uint {
        self.v.iter().filter(|elt| elt.is_some()).count()
    }

    /// Return `true` if there are no elements in the map.
    fn is_empty(&self) -> bool {
        self.v.iter().all(|elt| elt.is_none())
    }
}

impl<V> Mutable for SmallIntMap<V> {
    /// Clear the map, removing all key-value pairs.
    fn clear(&mut self) { self.v.clear() }
}

impl<V> Map<uint, V> for SmallIntMap<V> {
    /// Return a reference to the value corresponding to the key.
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
    /// Return a mutable reference to the value corresponding to the key.
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
    /// key is replaced by the new value. Return `true` if the key did
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

    /// Remove a key-value pair from the map. Return `true` if the key
    /// was present in the map, otherwise `false`.
    fn remove(&mut self, key: &uint) -> bool {
        self.pop(key).is_some()
    }

    /// Insert a key-value pair from the map. If the key already had a value
    /// present in the map, that value is returned. Otherwise `None` is returned.
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

impl<V:Clone> Clone for SmallIntMap<V> {
    #[inline]
    fn clone(&self) -> SmallIntMap<V> {
        SmallIntMap { v: self.v.clone() }
    }

    #[inline]
    fn clone_from(&mut self, source: &SmallIntMap<V>) {
        self.v.reserve(source.v.len());
        for (i, w) in self.v.mut_iter().enumerate() {
            *w = source.v[i].clone();
        }
    }
}

impl <S: hash::Writer, T: Hash<S>> Hash<S> for SmallIntMap<T> {
    fn hash(&self, state: &mut S) {
        self.v.hash(state)
    }
}

impl<V> SmallIntMap<V> {
    /// Create an empty SmallIntMap.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::SmallIntMap;
    /// let mut map: SmallIntMap<&str> = SmallIntMap::new();
    /// ```
    pub fn new() -> SmallIntMap<V> { SmallIntMap{v: vec!()} }

    /// Create an empty SmallIntMap with space for at least `capacity` elements
    /// before resizing.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::SmallIntMap;
    /// let mut map: SmallIntMap<&str> = SmallIntMap::with_capacity(10);
    /// ```
    pub fn with_capacity(capacity: uint) -> SmallIntMap<V> {
        SmallIntMap { v: Vec::with_capacity(capacity) }
    }

    /// Retrieves a value for the given key.
    /// See [`find`](../trait.Map.html#tymethod.find) for a non-failing alternative.
    ///
    /// # Failure
    ///
    /// Fails if the key is not present.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::SmallIntMap;
    ///
    /// let mut map = SmallIntMap::new();
    /// map.insert(1, "a");
    /// assert_eq!(map.get(&1), &"a");
    /// ```
    pub fn get<'a>(&'a self, key: &uint) -> &'a V {
        self.find(key).expect("key not present")
    }

    /// An iterator visiting all keys in ascending order by the keys.
    /// Iterator element type is `uint`.
    pub fn keys<'r>(&'r self) -> Keys<'r, V> {
        self.iter().map(|(k, _v)| k)
    }

    /// An iterator visiting all values in ascending order by the keys.
    /// Iterator element type is `&'r V`.
    pub fn values<'r>(&'r self) -> Values<'r, V> {
        self.iter().map(|(_k, v)| v)
    }

    /// An iterator visiting all key-value pairs in ascending order by the keys.
    /// Iterator element type is `(uint, &'r V)`.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::SmallIntMap;
    ///
    /// let mut map = SmallIntMap::new();
    /// map.insert(1, "a");
    /// map.insert(3, "c");
    /// map.insert(2, "b");
    ///
    /// // Print `1: a` then `2: b` then `3: c`
    /// for (key, value) in map.iter() {
    ///     println!("{}: {}", key, value);
    /// }
    /// ```
    pub fn iter<'r>(&'r self) -> Entries<'r, V> {
        Entries {
            front: 0,
            back: self.v.len(),
            iter: self.v.iter()
        }
    }

    /// An iterator visiting all key-value pairs in ascending order by the keys,
    /// with mutable references to the values
    /// Iterator element type is `(uint, &'r mut V)`.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::SmallIntMap;
    ///
    /// let mut map = SmallIntMap::new();
    /// map.insert(1, "a");
    /// map.insert(2, "b");
    /// map.insert(3, "c");
    ///
    /// for (key, value) in map.mut_iter() {
    ///     *value = "x";
    /// }
    ///
    /// for (key, value) in map.iter() {
    ///     assert_eq!(value, &"x");
    /// }
    /// ```
    pub fn mut_iter<'r>(&'r mut self) -> MutEntries<'r, V> {
        MutEntries {
            front: 0,
            back: self.v.len(),
            iter: self.v.mut_iter()
        }
    }

    /// Empties the map, moving all values into the specified closure.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::SmallIntMap;
    ///
    /// let mut map = SmallIntMap::new();
    /// map.insert(1, "a");
    /// map.insert(3, "c");
    /// map.insert(2, "b");
    ///
    /// // Not possible with .iter()
    /// let vec: Vec<(uint, &str)> = map.move_iter().collect();
    ///
    /// assert_eq!(vec, vec![(1, "a"), (2, "b"), (3, "c")]);
    /// ```
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
    /// Update a value in the map. If the key already exists in the map,
    /// modify the value with `ff` taking `oldval, newval`.
    /// Otherwise set the value to `newval`.
    /// Return `true` if the key did not already exist in the map.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::SmallIntMap;
    ///
    /// let mut map = SmallIntMap::new();
    ///
    /// // Key does not exist, will do a simple insert
    /// assert!(map.update(1, vec![1i, 2], |old, new| old.append(new.as_slice())));
    /// assert_eq!(map.get(&1), &vec![1i, 2]);
    ///
    /// // Key exists, update the value
    /// assert!(!map.update(1, vec![3i, 4], |old, new| old.append(new.as_slice())));
    /// assert_eq!(map.get(&1), &vec![1i, 2, 3, 4]);
    /// ```
    pub fn update(&mut self, key: uint, newval: V, ff: |V, V| -> V) -> bool {
        self.update_with_key(key, newval, |_k, v, v1| ff(v,v1))
    }

    /// Update a value in the map. If the key already exists in the map,
    /// modify the value with `ff` taking `key, oldval, newval`.
    /// Otherwise set the value to `newval`.
    /// Return `true` if the key did not already exist in the map.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::SmallIntMap;
    ///
    /// let mut map = SmallIntMap::new();
    ///
    /// // Key does not exist, will do a simple insert
    /// assert!(map.update_with_key(7, 10, |key, old, new| (old + new) % key));
    /// assert_eq!(map.get(&7), &10);
    ///
    /// // Key exists, update the value
    /// assert!(!map.update_with_key(7, 20, |key, old, new| (old + new) % key));
    /// assert_eq!(map.get(&7), &2);
    /// ```
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
}

impl<V: PartialOrd> PartialOrd for SmallIntMap<V> {
    #[inline]
    fn partial_cmp(&self, other: &SmallIntMap<V>) -> Option<Ordering> {
        iter::order::partial_cmp(self.iter(), other.iter())
    }
}

impl<V: Ord> Ord for SmallIntMap<V> {
    #[inline]
    fn cmp(&self, other: &SmallIntMap<V>) -> Ordering {
        iter::order::cmp(self.iter(), other.iter())
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

impl<V> FromIterator<(uint, V)> for SmallIntMap<V> {
    fn from_iter<Iter: Iterator<(uint, V)>>(iter: Iter) -> SmallIntMap<V> {
        let mut map = SmallIntMap::new();
        map.extend(iter);
        map
    }
}

impl<V> Extendable<(uint, V)> for SmallIntMap<V> {
    fn extend<Iter: Iterator<(uint, V)>>(&mut self, mut iter: Iter) {
        for (k, v) in iter {
            self.insert(k, v);
        }
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

/// Forward iterator over a map.
pub struct Entries<'a, T> {
    front: uint,
    back: uint,
    iter: slice::Items<'a, Option<T>>
}

iterator!(impl Entries -> (uint, &'a T), get_ref)
double_ended_iterator!(impl Entries -> (uint, &'a T), get_ref)

/// Forward iterator over the key-value pairs of a map, with the
/// values being mutable.
pub struct MutEntries<'a, T> {
    front: uint,
    back: uint,
    iter: slice::MutItems<'a, Option<T>>
}

iterator!(impl MutEntries -> (uint, &'a mut T), get_mut_ref)
double_ended_iterator!(impl MutEntries -> (uint, &'a mut T), get_mut_ref)

/// Forward iterator over the keys of a map
pub type Keys<'a, T> =
    iter::Map<'static, (uint, &'a T), uint, Entries<'a, T>>;

/// Forward iterator over the values of a map
pub type Values<'a, T> =
    iter::Map<'static, (uint, &'a T), &'a T, Entries<'a, T>>;

#[cfg(test)]
mod test_map {
    use std::prelude::*;
    use vec::Vec;
    use hash;

    use {Map, MutableMap, Mutable, MutableSeq};
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
    fn test_keys() {
        let mut map = SmallIntMap::new();
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
        let mut map = SmallIntMap::new();
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

    #[test]
    fn test_clone() {
        let mut a = SmallIntMap::new();

        a.insert(1, 'x');
        a.insert(4, 'y');
        a.insert(6, 'z');

        assert!(a.clone() == a);
    }

    #[test]
    fn test_eq() {
        let mut a = SmallIntMap::new();
        let mut b = SmallIntMap::new();

        assert!(a == b);
        assert!(a.insert(0, 5i));
        assert!(a != b);
        assert!(b.insert(0, 4i));
        assert!(a != b);
        assert!(a.insert(5, 19));
        assert!(a != b);
        assert!(!b.insert(0, 5));
        assert!(a != b);
        assert!(b.insert(5, 19));
        assert!(a == b);
    }

    #[test]
    fn test_lt() {
        let mut a = SmallIntMap::new();
        let mut b = SmallIntMap::new();

        assert!(!(a < b) && !(b < a));
        assert!(b.insert(2u, 5i));
        assert!(a < b);
        assert!(a.insert(2, 7));
        assert!(!(a < b) && b < a);
        assert!(b.insert(1, 0));
        assert!(b < a);
        assert!(a.insert(0, 6));
        assert!(a < b);
        assert!(a.insert(6, 2));
        assert!(a < b && !(b < a));
    }

    #[test]
    fn test_ord() {
        let mut a = SmallIntMap::new();
        let mut b = SmallIntMap::new();

        assert!(a <= b && a >= b);
        assert!(a.insert(1u, 1i));
        assert!(a > b && a >= b);
        assert!(b < a && b <= a);
        assert!(b.insert(2, 2));
        assert!(b > a && b >= a);
        assert!(a < b && a <= b);
    }

    #[test]
    fn test_hash() {
        let mut x = SmallIntMap::new();
        let mut y = SmallIntMap::new();

        assert!(hash::hash(&x) == hash::hash(&y));
        x.insert(1, 'a');
        x.insert(2, 'b');
        x.insert(3, 'c');

        y.insert(3, 'c');
        y.insert(2, 'b');
        y.insert(1, 'a');

        assert!(hash::hash(&x) == hash::hash(&y));
    }

    #[test]
    fn test_from_iter() {
        let xs: Vec<(uint, char)> = vec![(1u, 'a'), (2, 'b'), (3, 'c'), (4, 'd'), (5, 'e')];

        let map: SmallIntMap<char> = xs.iter().map(|&x| x).collect();

        for &(k, v) in xs.iter() {
            assert_eq!(map.find(&k), Some(&v));
        }
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
