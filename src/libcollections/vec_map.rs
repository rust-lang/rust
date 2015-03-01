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

use self::Entry::*;

use core::prelude::*;

use core::cmp::{max, Ordering};
use core::default::Default;
use core::fmt;
use core::hash::{Hash, Hasher};
use core::iter::{Enumerate, FilterMap, Map, FromIterator, IntoIterator};
use core::iter;
use core::mem::{replace, swap};
use core::ops::{Index, IndexMut};

use {vec, slice};
use vec::Vec;

/// A map optimized for small integer keys.
///
/// # Examples
///
/// ```
/// # #![feature(collections)]
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
pub struct VecMap<V> {
    v: Vec<Option<V>>,
}

/// A view into a single entry in a map, which may either be vacant or occupied.

#[stable(feature = "rust1", since = "1.0.0")]
pub enum Entry<'a, V:'a> {
    /// A vacant Entry
    #[stable(feature = "rust1", since = "1.0.0")]
    Vacant(VacantEntry<'a, V>),

    /// An occupied Entry
    #[stable(feature = "rust1", since = "1.0.0")]
    Occupied(OccupiedEntry<'a, V>),
}

/// A vacant Entry.

#[stable(feature = "rust1", since = "1.0.0")]
pub struct VacantEntry<'a, V:'a> {
    map: &'a mut VecMap<V>,
    index: usize,
}

/// An occupied Entry.
#[stable(feature = "rust1", since = "1.0.0")]
pub struct OccupiedEntry<'a, V:'a> {
    map: &'a mut VecMap<V>,
    index: usize,
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<V> Default for VecMap<V> {
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    fn default() -> VecMap<V> { VecMap::new() }
}

#[stable(feature = "rust1", since = "1.0.0")]
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

#[stable(feature = "rust1", since = "1.0.0")]
impl<V: Hash> Hash for VecMap<V> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // In order to not traverse the `VecMap` twice, count the elements
        // during iteration.
        let mut count: usize = 0;
        for elt in self {
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
    /// # #![feature(collections)]
    /// use std::collections::VecMap;
    /// let mut map: VecMap<&str> = VecMap::new();
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn new() -> VecMap<V> { VecMap { v: vec![] } }

    /// Creates an empty `VecMap` with space for at least `capacity`
    /// elements before resizing.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(collections)]
    /// use std::collections::VecMap;
    /// let mut map: VecMap<&str> = VecMap::with_capacity(10);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn with_capacity(capacity: usize) -> VecMap<V> {
        VecMap { v: Vec::with_capacity(capacity) }
    }

    /// Returns the number of elements the `VecMap` can hold without
    /// reallocating.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(collections)]
    /// use std::collections::VecMap;
    /// let map: VecMap<String> = VecMap::with_capacity(10);
    /// assert!(map.capacity() >= 10);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn capacity(&self) -> usize {
        self.v.capacity()
    }

    /// Reserves capacity for the given `VecMap` to contain `len` distinct keys.
    /// In the case of `VecMap` this means reallocations will not occur as long
    /// as all inserted keys are less than `len`.
    ///
    /// The collection may reserve more space to avoid frequent reallocations.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(collections)]
    /// use std::collections::VecMap;
    /// let mut map: VecMap<&str> = VecMap::new();
    /// map.reserve_len(10);
    /// assert!(map.capacity() >= 10);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn reserve_len(&mut self, len: usize) {
        let cur_len = self.v.len();
        if len >= cur_len {
            self.v.reserve(len - cur_len);
        }
    }

    /// Reserves the minimum capacity for the given `VecMap` to contain `len` distinct keys.
    /// In the case of `VecMap` this means reallocations will not occur as long as all inserted
    /// keys are less than `len`.
    ///
    /// Note that the allocator may give the collection more space than it requests.
    /// Therefore capacity cannot be relied upon to be precisely minimal.  Prefer
    /// `reserve_len` if future insertions are expected.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(collections)]
    /// use std::collections::VecMap;
    /// let mut map: VecMap<&str> = VecMap::new();
    /// map.reserve_len_exact(10);
    /// assert!(map.capacity() >= 10);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn reserve_len_exact(&mut self, len: usize) {
        let cur_len = self.v.len();
        if len >= cur_len {
            self.v.reserve_exact(len - cur_len);
        }
    }

    /// Returns an iterator visiting all keys in ascending order of the keys.
    /// The iterator's element type is `usize`.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn keys<'r>(&'r self) -> Keys<'r, V> {
        fn first<A, B>((a, _): (A, B)) -> A { a }
        let first: fn((usize, &'r V)) -> usize = first; // coerce to fn pointer

        Keys { iter: self.iter().map(first) }
    }

    /// Returns an iterator visiting all values in ascending order of the keys.
    /// The iterator's element type is `&'r V`.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn values<'r>(&'r self) -> Values<'r, V> {
        fn second<A, B>((_, b): (A, B)) -> B { b }
        let second: fn((usize, &'r V)) -> &'r V = second; // coerce to fn pointer

        Values { iter: self.iter().map(second) }
    }

    /// Returns an iterator visiting all key-value pairs in ascending order of the keys.
    /// The iterator's element type is `(usize, &'r V)`.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(collections)]
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
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn iter<'r>(&'r self) -> Iter<'r, V> {
        Iter {
            front: 0,
            back: self.v.len(),
            iter: self.v.iter()
        }
    }

    /// Returns an iterator visiting all key-value pairs in ascending order of the keys,
    /// with mutable references to the values.
    /// The iterator's element type is `(usize, &'r mut V)`.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(collections)]
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
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn iter_mut<'r>(&'r mut self) -> IterMut<'r, V> {
        IterMut {
            front: 0,
            back: self.v.len(),
            iter: self.v.iter_mut()
        }
    }

    /// Returns an iterator visiting all key-value pairs in ascending order of
    /// the keys, consuming the original `VecMap`.
    /// The iterator's element type is `(usize, &'r V)`.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(collections)]
    /// use std::collections::VecMap;
    ///
    /// let mut map = VecMap::new();
    /// map.insert(1, "a");
    /// map.insert(3, "c");
    /// map.insert(2, "b");
    ///
    /// let vec: Vec<(usize, &str)> = map.into_iter().collect();
    ///
    /// assert_eq!(vec, [(1, "a"), (2, "b"), (3, "c")]);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn into_iter(self) -> IntoIter<V> {
        fn filter<A>((i, v): (usize, Option<A>)) -> Option<(usize, A)> {
            v.map(|v| (i, v))
        }
        let filter: fn((usize, Option<V>)) -> Option<(usize, V)> = filter; // coerce to fn ptr

        IntoIter { iter: self.v.into_iter().enumerate().filter_map(filter) }
    }

    /// Moves all elements from `other` into the map while overwriting existing keys.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(collections)]
    /// use std::collections::VecMap;
    ///
    /// let mut a = VecMap::new();
    /// a.insert(1, "a");
    /// a.insert(2, "b");
    ///
    /// let mut b = VecMap::new();
    /// b.insert(3, "c");
    /// b.insert(4, "d");
    ///
    /// a.append(&mut b);
    ///
    /// assert_eq!(a.len(), 4);
    /// assert_eq!(b.len(), 0);
    /// assert_eq!(a[1], "a");
    /// assert_eq!(a[2], "b");
    /// assert_eq!(a[3], "c");
    /// assert_eq!(a[4], "d");
    /// ```
    #[unstable(feature = "collections",
               reason = "recently added as part of collections reform 2")]
    pub fn append(&mut self, other: &mut Self) {
        self.extend(other.drain());
    }

    /// Splits the collection into two at the given key.
    ///
    /// Returns a newly allocated `Self`. `self` contains elements `[0, at)`,
    /// and the returned `Self` contains elements `[at, max_key)`.
    ///
    /// Note that the capacity of `self` does not change.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(collections)]
    /// use std::collections::VecMap;
    ///
    /// let mut a = VecMap::new();
    /// a.insert(1, "a");
    /// a.insert(2, "b");
    /// a.insert(3, "c");
    /// a.insert(4, "d");
    ///
    /// let b = a.split_off(3);
    ///
    /// assert_eq!(a[1], "a");
    /// assert_eq!(a[2], "b");
    ///
    /// assert_eq!(b[3], "c");
    /// assert_eq!(b[4], "d");
    /// ```
    #[unstable(feature = "collections",
               reason = "recently added as part of collections reform 2")]
    pub fn split_off(&mut self, at: usize) -> Self {
        let mut other = VecMap::new();

        if at == 0 {
            // Move all elements to other
            swap(self, &mut other);
            return other
        } else if at > self.v.len() {
            // No elements to copy
            return other;
        }

        // Look up the index of the first non-None item
        let first_index = self.v.iter().position(|el| el.is_some());
        let start_index = match first_index {
            Some(index) => max(at, index),
            None => {
                // self has no elements
                return other;
            }
        };

        // Fill the new VecMap with `None`s until `start_index`
        other.v.extend((0..start_index).map(|_| None));

        // Move elements beginning with `start_index` from `self` into `other`
        other.v.extend(self.v[start_index..].iter_mut().map(|el| el.take()));

        other
    }

    /// Returns an iterator visiting all key-value pairs in ascending order of
    /// the keys, emptying (but not consuming) the original `VecMap`.
    /// The iterator's element type is `(usize, &'r V)`. Keeps the allocated memory for reuse.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(collections)]
    /// use std::collections::VecMap;
    ///
    /// let mut map = VecMap::new();
    /// map.insert(1, "a");
    /// map.insert(3, "c");
    /// map.insert(2, "b");
    ///
    /// let vec: Vec<(usize, &str)> = map.drain().collect();
    ///
    /// assert_eq!(vec, [(1, "a"), (2, "b"), (3, "c")]);
    /// ```
    #[unstable(feature = "collections",
               reason = "matches collection reform specification, waiting for dust to settle")]
    pub fn drain<'a>(&'a mut self) -> Drain<'a, V> {
        fn filter<A>((i, v): (usize, Option<A>)) -> Option<(usize, A)> {
            v.map(|v| (i, v))
        }
        let filter: fn((usize, Option<V>)) -> Option<(usize, V)> = filter; // coerce to fn ptr

        Drain { iter: self.v.drain().enumerate().filter_map(filter) }
    }

    /// Return the number of elements in the map.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(collections)]
    /// use std::collections::VecMap;
    ///
    /// let mut a = VecMap::new();
    /// assert_eq!(a.len(), 0);
    /// a.insert(1, "a");
    /// assert_eq!(a.len(), 1);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn len(&self) -> usize {
        self.v.iter().filter(|elt| elt.is_some()).count()
    }

    /// Return true if the map contains no elements.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(collections)]
    /// use std::collections::VecMap;
    ///
    /// let mut a = VecMap::new();
    /// assert!(a.is_empty());
    /// a.insert(1, "a");
    /// assert!(!a.is_empty());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn is_empty(&self) -> bool {
        self.v.iter().all(|elt| elt.is_none())
    }

    /// Clears the map, removing all key-value pairs.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(collections)]
    /// use std::collections::VecMap;
    ///
    /// let mut a = VecMap::new();
    /// a.insert(1, "a");
    /// a.clear();
    /// assert!(a.is_empty());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn clear(&mut self) { self.v.clear() }

    /// Returns a reference to the value corresponding to the key.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(collections)]
    /// use std::collections::VecMap;
    ///
    /// let mut map = VecMap::new();
    /// map.insert(1, "a");
    /// assert_eq!(map.get(&1), Some(&"a"));
    /// assert_eq!(map.get(&2), None);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn get(&self, key: &usize) -> Option<&V> {
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
    /// # #![feature(collections)]
    /// use std::collections::VecMap;
    ///
    /// let mut map = VecMap::new();
    /// map.insert(1, "a");
    /// assert_eq!(map.contains_key(&1), true);
    /// assert_eq!(map.contains_key(&2), false);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn contains_key(&self, key: &usize) -> bool {
        self.get(key).is_some()
    }

    /// Returns a mutable reference to the value corresponding to the key.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(collections)]
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
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn get_mut(&mut self, key: &usize) -> Option<&mut V> {
        if *key < self.v.len() {
            match *(&mut self.v[*key]) {
              Some(ref mut value) => Some(value),
              None => None
            }
        } else {
            None
        }
    }

    /// Inserts a key-value pair from the map. If the key already had a value
    /// present in the map, that value is returned. Otherwise, `None` is returned.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(collections)]
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
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn insert(&mut self, key: usize, value: V) -> Option<V> {
        let len = self.v.len();
        if len <= key {
            self.v.extend((0..key - len + 1).map(|_| None));
        }
        replace(&mut self.v[key], Some(value))
    }

    /// Removes a key from the map, returning the value at the key if the key
    /// was previously in the map.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(collections)]
    /// use std::collections::VecMap;
    ///
    /// let mut map = VecMap::new();
    /// map.insert(1, "a");
    /// assert_eq!(map.remove(&1), Some("a"));
    /// assert_eq!(map.remove(&1), None);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn remove(&mut self, key: &usize) -> Option<V> {
        if *key >= self.v.len() {
            return None;
        }
        let result = &mut self.v[*key];
        result.take()
    }

    /// Gets the given key's corresponding entry in the map for in-place manipulation.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(collections)]
    /// use std::collections::VecMap;
    /// use std::collections::vec_map::Entry;
    ///
    /// let mut count: VecMap<u32> = VecMap::new();
    ///
    /// // count the number of occurrences of numbers in the vec
    /// for x in vec![1, 2, 1, 2, 3, 4, 1, 2, 4].iter() {
    ///     match count.entry(*x) {
    ///         Entry::Vacant(view) => {
    ///             view.insert(1);
    ///         },
    ///         Entry::Occupied(mut view) => {
    ///             let v = view.get_mut();
    ///             *v += 1;
    ///         },
    ///     }
    /// }
    ///
    /// assert_eq!(count[1], 3);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn entry(&mut self, key: usize) -> Entry<V> {
        // FIXME(Gankro): this is basically the dumbest implementation of
        // entry possible, because weird non-lexical borrows issues make it
        // completely insane to do any other way. That said, Entry is a border-line
        // useless construct on VecMap, so it's hardly a big loss.
        if self.contains_key(&key) {
            Occupied(OccupiedEntry {
                map: self,
                index: key,
            })
        } else {
            Vacant(VacantEntry {
                map: self,
                index: key,
            })
        }
    }
}


impl<'a, V> Entry<'a, V> {
    #[unstable(feature = "collections",
               reason = "will soon be replaced by or_insert")]
    #[deprecated(since = "1.0",
                reason = "replaced with more ergonomic `default` and `default_with`")]
    /// Returns a mutable reference to the entry if occupied, or the VacantEntry if vacant
    pub fn get(self) -> Result<&'a mut V, VacantEntry<'a, V>> {
        match self {
            Occupied(entry) => Ok(entry.into_mut()),
            Vacant(entry) => Err(entry),
        }
    }

    #[unstable(feature = "collections",
               reason = "matches entry v3 specification, waiting for dust to settle")]
    /// Ensures a value is in the entry by inserting the default if empty, and returns
    /// a mutable reference to the value in the entry.
    pub fn default(self, default: V) -> &'a mut V {
        match self {
            Occupied(entry) => entry.into_mut(),
            Vacant(entry) => entry.insert(default),
        }
    }

    #[unstable(feature = "collections",
               reason = "matches entry v3 specification, waiting for dust to settle")]
    /// Ensures a value is in the entry by inserting the result of the default function if empty,
    /// and returns a mutable reference to the value in the entry.
    pub fn default_with<F: FnOnce() -> V>(self, default: F) -> &'a mut V {
        match self {
            Occupied(entry) => entry.into_mut(),
            Vacant(entry) => entry.insert(default()),
        }
    }
}

impl<'a, V> VacantEntry<'a, V> {
    /// Sets the value of the entry with the VacantEntry's key,
    /// and returns a mutable reference to it.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn insert(self, value: V) -> &'a mut V {
        let index = self.index;
        self.map.insert(index, value);
        &mut self.map[index]
    }
}

impl<'a, V> OccupiedEntry<'a, V> {
    /// Gets a reference to the value in the entry.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn get(&self) -> &V {
        let index = self.index;
        &self.map[index]
    }

    /// Gets a mutable reference to the value in the entry.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn get_mut(&mut self) -> &mut V {
        let index = self.index;
        &mut self.map[index]
    }

    /// Converts the entry into a mutable reference to its value.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn into_mut(self) -> &'a mut V {
        let index = self.index;
        &mut self.map[index]
    }

    /// Sets the value of the entry with the OccupiedEntry's key,
    /// and returns the entry's old value.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn insert(&mut self, value: V) -> V {
        let index = self.index;
        self.map.insert(index, value).unwrap()
    }

    /// Takes the value of the entry out of the map, and returns it.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn remove(self) -> V {
        let index = self.index;
        self.map.remove(&index).unwrap()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<V: PartialEq> PartialEq for VecMap<V> {
    fn eq(&self, other: &VecMap<V>) -> bool {
        iter::order::eq(self.iter(), other.iter())
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<V: Eq> Eq for VecMap<V> {}

#[stable(feature = "rust1", since = "1.0.0")]
impl<V: PartialOrd> PartialOrd for VecMap<V> {
    #[inline]
    fn partial_cmp(&self, other: &VecMap<V>) -> Option<Ordering> {
        iter::order::partial_cmp(self.iter(), other.iter())
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<V: Ord> Ord for VecMap<V> {
    #[inline]
    fn cmp(&self, other: &VecMap<V>) -> Ordering {
        iter::order::cmp(self.iter(), other.iter())
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<V: fmt::Debug> fmt::Debug for VecMap<V> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        try!(write!(f, "{{"));

        for (i, (k, v)) in self.iter().enumerate() {
            if i != 0 { try!(write!(f, ", ")); }
            try!(write!(f, "{}: {:?}", k, *v));
        }

        write!(f, "}}")
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<V> FromIterator<(usize, V)> for VecMap<V> {
    fn from_iter<I: IntoIterator<Item=(usize, V)>>(iter: I) -> VecMap<V> {
        let mut map = VecMap::new();
        map.extend(iter);
        map
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> IntoIterator for VecMap<T> {
    type Item = (usize, T);
    type IntoIter = IntoIter<T>;

    fn into_iter(self) -> IntoIter<T> {
        self.into_iter()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> IntoIterator for &'a VecMap<T> {
    type Item = (usize, &'a T);
    type IntoIter = Iter<'a, T>;

    fn into_iter(self) -> Iter<'a, T> {
        self.iter()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> IntoIterator for &'a mut VecMap<T> {
    type Item = (usize, &'a mut T);
    type IntoIter = IterMut<'a, T>;

    fn into_iter(mut self) -> IterMut<'a, T> {
        self.iter_mut()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<V> Extend<(usize, V)> for VecMap<V> {
    fn extend<I: IntoIterator<Item=(usize, V)>>(&mut self, iter: I) {
        for (k, v) in iter {
            self.insert(k, v);
        }
    }
}

#[cfg(stage0)]
impl<V> Index<usize> for VecMap<V> {
    type Output = V;

    #[inline]
    fn index<'a>(&'a self, i: &usize) -> &'a V {
        self.get(i).expect("key not present")
    }
}

#[cfg(not(stage0))]
impl<V> Index<usize> for VecMap<V> {
    type Output = V;

    #[inline]
    fn index<'a>(&'a self, i: usize) -> &'a V {
        self.get(&i).expect("key not present")
    }
}

#[cfg(not(stage0))]
impl<'a,V> Index<&'a usize> for VecMap<V> {
    type Output = V;

    #[inline]
    fn index(&self, i: &usize) -> &V {
        self.get(i).expect("key not present")
    }
}

#[cfg(stage0)]
#[stable(feature = "rust1", since = "1.0.0")]
impl<V> IndexMut<usize> for VecMap<V> {
    #[inline]
    fn index_mut(&mut self, i: &usize) -> &mut V {
        self.get_mut(&i).expect("key not present")
    }
}

#[cfg(not(stage0))]
#[stable(feature = "rust1", since = "1.0.0")]
impl<V> IndexMut<usize> for VecMap<V> {
    #[inline]
    fn index_mut(&mut self, i: usize) -> &mut V {
        self.get_mut(&i).expect("key not present")
    }
}

#[cfg(not(stage0))]
#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, V> IndexMut<&'a usize> for VecMap<V> {
    #[inline]
    fn index_mut(&mut self, i: &usize) -> &mut V {
        self.get_mut(i).expect("key not present")
    }
}

macro_rules! iterator {
    (impl $name:ident -> $elem:ty, $($getter:ident),+) => {
        #[stable(feature = "rust1", since = "1.0.0")]
        impl<'a, V> Iterator for $name<'a, V> {
            type Item = $elem;

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
            fn size_hint(&self) -> (usize, Option<usize>) {
                (0, Some(self.back - self.front))
            }
        }
    }
}

macro_rules! double_ended_iterator {
    (impl $name:ident -> $elem:ty, $($getter:ident),+) => {
        #[stable(feature = "rust1", since = "1.0.0")]
        impl<'a, V> DoubleEndedIterator for $name<'a, V> {
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
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Iter<'a, V:'a> {
    front: usize,
    back: usize,
    iter: slice::Iter<'a, Option<V>>
}

// FIXME(#19839) Remove in favor of `#[derive(Clone)]`
impl<'a, V> Clone for Iter<'a, V> {
    fn clone(&self) -> Iter<'a, V> {
        Iter {
            front: self.front,
            back: self.back,
            iter: self.iter.clone()
        }
    }
}

iterator! { impl Iter -> (usize, &'a V), as_ref }
double_ended_iterator! { impl Iter -> (usize, &'a V), as_ref }

/// An iterator over the key-value pairs of a map, with the
/// values being mutable.
#[stable(feature = "rust1", since = "1.0.0")]
pub struct IterMut<'a, V:'a> {
    front: usize,
    back: usize,
    iter: slice::IterMut<'a, Option<V>>
}

iterator! { impl IterMut -> (usize, &'a mut V), as_mut }
double_ended_iterator! { impl IterMut -> (usize, &'a mut V), as_mut }

/// An iterator over the keys of a map.
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Keys<'a, V: 'a> {
    iter: Map<Iter<'a, V>, fn((usize, &'a V)) -> usize>
}

// FIXME(#19839) Remove in favor of `#[derive(Clone)]`
impl<'a, V> Clone for Keys<'a, V> {
    fn clone(&self) -> Keys<'a, V> {
        Keys {
            iter: self.iter.clone()
        }
    }
}

/// An iterator over the values of a map.
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Values<'a, V: 'a> {
    iter: Map<Iter<'a, V>, fn((usize, &'a V)) -> &'a V>
}

// FIXME(#19839) Remove in favor of `#[derive(Clone)]`
impl<'a, V> Clone for Values<'a, V> {
    fn clone(&self) -> Values<'a, V> {
        Values {
            iter: self.iter.clone()
        }
    }
}

/// A consuming iterator over the key-value pairs of a map.
#[stable(feature = "rust1", since = "1.0.0")]
pub struct IntoIter<V> {
    iter: FilterMap<
    Enumerate<vec::IntoIter<Option<V>>>,
    fn((usize, Option<V>)) -> Option<(usize, V)>>
}

#[unstable(feature = "collections")]
pub struct Drain<'a, V:'a> {
    iter: FilterMap<
    Enumerate<vec::Drain<'a, Option<V>>>,
    fn((usize, Option<V>)) -> Option<(usize, V)>>
}

#[unstable(feature = "collections")]
impl<'a, V> Iterator for Drain<'a, V> {
    type Item = (usize, V);

    fn next(&mut self) -> Option<(usize, V)> { self.iter.next() }
    fn size_hint(&self) -> (usize, Option<usize>) { self.iter.size_hint() }
}

#[unstable(feature = "collections")]
impl<'a, V> DoubleEndedIterator for Drain<'a, V> {
    fn next_back(&mut self) -> Option<(usize, V)> { self.iter.next_back() }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, V> Iterator for Keys<'a, V> {
    type Item = usize;

    fn next(&mut self) -> Option<usize> { self.iter.next() }
    fn size_hint(&self) -> (usize, Option<usize>) { self.iter.size_hint() }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, V> DoubleEndedIterator for Keys<'a, V> {
    fn next_back(&mut self) -> Option<usize> { self.iter.next_back() }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, V> Iterator for Values<'a, V> {
    type Item = &'a V;

    fn next(&mut self) -> Option<(&'a V)> { self.iter.next() }
    fn size_hint(&self) -> (usize, Option<usize>) { self.iter.size_hint() }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, V> DoubleEndedIterator for Values<'a, V> {
    fn next_back(&mut self) -> Option<(&'a V)> { self.iter.next_back() }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<V> Iterator for IntoIter<V> {
    type Item = (usize, V);

    fn next(&mut self) -> Option<(usize, V)> { self.iter.next() }
    fn size_hint(&self) -> (usize, Option<usize>) { self.iter.size_hint() }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl<V> DoubleEndedIterator for IntoIter<V> {
    fn next_back(&mut self) -> Option<(usize, V)> { self.iter.next_back() }
}
