// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Collection types.
//!
//! See [../std/collections](std::collections) for a detailed discussion of collections in Rust.


#![crate_name = "collections"]
#![experimental]
#![crate_type = "rlib"]
#![license = "MIT/ASL2"]
#![doc(html_logo_url = "http://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
       html_favicon_url = "http://www.rust-lang.org/favicon.ico",
       html_root_url = "http://doc.rust-lang.org/nightly/",
       html_playground_url = "http://play.rust-lang.org/")]

#![allow(unknown_features)]
#![feature(macro_rules, default_type_params, phase, globs)]
#![feature(unsafe_destructor, import_shadowing, slicing_syntax)]
#![no_std]

#[phase(plugin, link)] extern crate core;
extern crate unicode;
extern crate alloc;

#[cfg(test)] extern crate native;
#[cfg(test)] extern crate test;

#[cfg(test)] #[phase(plugin, link)] extern crate std;
#[cfg(test)] #[phase(plugin, link)] extern crate log;

use core::prelude::Option;

pub use bitv::{Bitv, BitvSet};
pub use btree::{BTreeMap, BTreeSet};
pub use core::prelude::Collection;
pub use dlist::DList;
pub use enum_set::EnumSet;
pub use priority_queue::PriorityQueue;
pub use ringbuf::RingBuf;
pub use smallintmap::SmallIntMap;
pub use string::String;
pub use treemap::{TreeMap, TreeSet};
pub use trie::{TrieMap, TrieSet};
pub use vec::Vec;

mod macros;

pub mod bitv;
pub mod btree;
pub mod dlist;
pub mod enum_set;
pub mod priority_queue;
pub mod ringbuf;
pub mod smallintmap;
pub mod treemap;
pub mod trie;
pub mod slice;
pub mod str;
pub mod string;
pub mod vec;
pub mod hash;

mod deque;

/// A mutable container type.
pub trait Mutable: Collection {
    /// Clears the container, removing all values.
    ///
    /// # Example
    ///
    /// ```
    /// let mut v = vec![1i, 2, 3];
    /// v.clear();
    /// assert!(v.is_empty());
    /// ```
    fn clear(&mut self);
}

/// A key-value store where values may be looked up by their keys. This trait
/// provides basic operations to operate on these stores.
pub trait Map<K, V>: Collection {
    /// Returns a reference to the value corresponding to the key.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::HashMap;
    ///
    /// let mut map = HashMap::new();
    /// map.insert("a", 1i);
    /// assert_eq!(map.find(&"a"), Some(&1i));
    /// assert_eq!(map.find(&"b"), None);
    /// ```
    fn find<'a>(&'a self, key: &K) -> Option<&'a V>;

    /// Returns true if the map contains a value for the specified key.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::HashMap;
    ///
    /// let mut map = HashMap::new();
    /// map.insert("a", 1i);
    /// assert_eq!(map.contains_key(&"a"), true);
    /// assert_eq!(map.contains_key(&"b"), false);
    /// ```
    #[inline]
    fn contains_key(&self, key: &K) -> bool {
        self.find(key).is_some()
    }
}

/// A key-value store (map) where the values can be modified.
pub trait MutableMap<K, V>: Map<K, V> + Mutable {
    /// Inserts a key-value pair into the map. An existing value for a
    /// key is replaced by the new value. Returns `true` if the key did
    /// not already exist in the map.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::HashMap;
    ///
    /// let mut map = HashMap::new();
    /// assert_eq!(map.insert("key", 2i), true);
    /// assert_eq!(map.insert("key", 9i), false);
    /// assert_eq!(map["key"], 9i);
    /// ```
    #[inline]
    fn insert(&mut self, key: K, value: V) -> bool {
        self.swap(key, value).is_none()
    }

    /// Removes a key-value pair from the map. Returns `true` if the key
    /// was present in the map.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::HashMap;
    ///
    /// let mut map = HashMap::new();
    /// assert_eq!(map.remove(&"key"), false);
    /// map.insert("key", 2i);
    /// assert_eq!(map.remove(&"key"), true);
    /// ```
    #[inline]
    fn remove(&mut self, key: &K) -> bool {
        self.pop(key).is_some()
    }

    /// Inserts a key-value pair into the map. If the key already had a value
    /// present in the map, that value is returned. Otherwise, `None` is
    /// returned.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::HashMap;
    ///
    /// let mut map = HashMap::new();
    /// assert_eq!(map.swap("a", 37i), None);
    /// assert_eq!(map.is_empty(), false);
    ///
    /// map.insert("a", 1i);
    /// assert_eq!(map.swap("a", 37i), Some(1i));
    /// assert_eq!(map["a"], 37i);
    /// ```
    fn swap(&mut self, k: K, v: V) -> Option<V>;

    /// Removes a key from the map, returning the value at the key if the key
    /// was previously in the map.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::HashMap;
    ///
    /// let mut map: HashMap<&str, int> = HashMap::new();
    /// map.insert("a", 1i);
    /// assert_eq!(map.pop(&"a"), Some(1i));
    /// assert_eq!(map.pop(&"a"), None);
    /// ```
    fn pop(&mut self, k: &K) -> Option<V>;

    /// Returns a mutable reference to the value corresponding to the key.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::HashMap;
    ///
    /// let mut map = HashMap::new();
    /// map.insert("a", 1i);
    /// match map.find_mut(&"a") {
    ///     Some(x) => *x = 7i,
    ///     None => (),
    /// }
    /// assert_eq!(map["a"], 7i);
    /// ```
    fn find_mut<'a>(&'a mut self, key: &K) -> Option<&'a mut V>;
}

/// A group of objects which are each distinct from one another. This
/// trait represents actions which can be performed on sets to iterate over
/// them.
pub trait Set<T>: Collection {
    /// Returns `true` if the set contains a value.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::HashSet;
    ///
    /// let set: HashSet<int> = [1i, 2, 3].iter().map(|&x| x).collect();
    /// assert_eq!(set.contains(&1), true);
    /// assert_eq!(set.contains(&4), false);
    /// ```
    fn contains(&self, value: &T) -> bool;

    /// Returns `true` if the set has no elements in common with `other`.
    /// This is equivalent to checking for an empty intersection.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::HashSet;
    ///
    /// let a: HashSet<int> = [1i, 2, 3].iter().map(|&x| x).collect();
    /// let mut b: HashSet<int> = HashSet::new();
    ///
    /// assert_eq!(a.is_disjoint(&b), true);
    /// b.insert(4);
    /// assert_eq!(a.is_disjoint(&b), true);
    /// b.insert(1);
    /// assert_eq!(a.is_disjoint(&b), false);
    /// ```
    fn is_disjoint(&self, other: &Self) -> bool;

    /// Returns `true` if the set is a subset of another.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::HashSet;
    ///
    /// let sup: HashSet<int> = [1i, 2, 3].iter().map(|&x| x).collect();
    /// let mut set: HashSet<int> = HashSet::new();
    ///
    /// assert_eq!(set.is_subset(&sup), true);
    /// set.insert(2);
    /// assert_eq!(set.is_subset(&sup), true);
    /// set.insert(4);
    /// assert_eq!(set.is_subset(&sup), false);
    /// ```
    fn is_subset(&self, other: &Self) -> bool;

    /// Returns `true` if the set is a superset of another.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::HashSet;
    ///
    /// let sub: HashSet<int> = [1i, 2].iter().map(|&x| x).collect();
    /// let mut set: HashSet<int> = HashSet::new();
    ///
    /// assert_eq!(set.is_superset(&sub), false);
    ///
    /// set.insert(0);
    /// set.insert(1);
    /// assert_eq!(set.is_superset(&sub), false);
    ///
    /// set.insert(2);
    /// assert_eq!(set.is_superset(&sub), true);
    /// ```
    fn is_superset(&self, other: &Self) -> bool {
        other.is_subset(self)
    }

    // FIXME #8154: Add difference, sym. difference, intersection and union iterators
}

/// A mutable collection of values which are distinct from one another that
/// can be mutated.
pub trait MutableSet<T>: Set<T> + Mutable {
    /// Adds a value to the set. Returns `true` if the value was not already
    /// present in the set.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::HashSet;
    ///
    /// let mut set = HashSet::new();
    ///
    /// assert_eq!(set.insert(2i), true);
    /// assert_eq!(set.insert(2i), false);
    /// assert_eq!(set.len(), 1);
    /// ```
    fn insert(&mut self, value: T) -> bool;

    /// Removes a value from the set. Returns `true` if the value was
    /// present in the set.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::HashSet;
    ///
    /// let mut set = HashSet::new();
    ///
    /// set.insert(2i);
    /// assert_eq!(set.remove(&2), true);
    /// assert_eq!(set.remove(&2), false);
    /// ```
    fn remove(&mut self, value: &T) -> bool;
}

pub trait MutableSeq<T>: Mutable {
    /// Appends an element to the back of a collection.
    ///
    /// # Example
    ///
    /// ```rust
    /// let mut vec = vec!(1i, 2);
    /// vec.push(3);
    /// assert_eq!(vec, vec!(1, 2, 3));
    /// ```
    fn push(&mut self, t: T);

    /// Removes the last element from a collection and returns it, or `None` if
    /// it is empty.
    ///
    /// # Example
    ///
    /// ```rust
    /// let mut vec = vec!(1i, 2, 3);
    /// assert_eq!(vec.pop(), Some(3));
    /// assert_eq!(vec, vec!(1, 2));
    /// ```
    fn pop(&mut self) -> Option<T>;
}

/// A double-ended sequence that allows querying, insertion and deletion at both
/// ends.
///
/// # Example
///
/// With a `Deque` we can simulate a queue efficiently:
///
/// ```
/// use std::collections::{RingBuf, Deque};
///
/// let mut queue = RingBuf::new();
/// queue.push(1i);
/// queue.push(2i);
/// queue.push(3i);
///
/// // Will print 1, 2, 3
/// while !queue.is_empty() {
///     let x = queue.pop_front().unwrap();
///     println!("{}", x);
/// }
/// ```
///
/// We can also simulate a stack:
///
/// ```
/// use std::collections::{RingBuf, Deque};
///
/// let mut stack = RingBuf::new();
/// stack.push_front(1i);
/// stack.push_front(2i);
/// stack.push_front(3i);
///
/// // Will print 3, 2, 1
/// while !stack.is_empty() {
///     let x = stack.pop_front().unwrap();
///     println!("{}", x);
/// }
/// ```
///
/// And of course we can mix and match:
///
/// ```
/// use std::collections::{DList, Deque};
///
/// let mut deque = DList::new();
///
/// // Init deque with 1, 2, 3, 4
/// deque.push_front(2i);
/// deque.push_front(1i);
/// deque.push(3i);
/// deque.push(4i);
///
/// // Will print (1, 4) and (2, 3)
/// while !deque.is_empty() {
///     let f = deque.pop_front().unwrap();
///     let b = deque.pop().unwrap();
///     println!("{}", (f, b));
/// }
/// ```
pub trait Deque<T> : MutableSeq<T> {
    /// Provides a reference to the front element, or `None` if the sequence is
    /// empty.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::{RingBuf, Deque};
    ///
    /// let mut d = RingBuf::new();
    /// assert_eq!(d.front(), None);
    ///
    /// d.push(1i);
    /// d.push(2i);
    /// assert_eq!(d.front(), Some(&1i));
    /// ```
    fn front<'a>(&'a self) -> Option<&'a T>;

    /// Provides a mutable reference to the front element, or `None` if the
    /// sequence is empty.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::{RingBuf, Deque};
    ///
    /// let mut d = RingBuf::new();
    /// assert_eq!(d.front_mut(), None);
    ///
    /// d.push(1i);
    /// d.push(2i);
    /// match d.front_mut() {
    ///     Some(x) => *x = 9i,
    ///     None => (),
    /// }
    /// assert_eq!(d.front(), Some(&9i));
    /// ```
    fn front_mut<'a>(&'a mut self) -> Option<&'a mut T>;

    /// Provides a reference to the back element, or `None` if the sequence is
    /// empty.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::{DList, Deque};
    ///
    /// let mut d = DList::new();
    /// assert_eq!(d.back(), None);
    ///
    /// d.push(1i);
    /// d.push(2i);
    /// assert_eq!(d.back(), Some(&2i));
    /// ```
    fn back<'a>(&'a self) -> Option<&'a T>;

    /// Provides a mutable reference to the back element, or `None` if the
    /// sequence is empty.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::{DList, Deque};
    ///
    /// let mut d = DList::new();
    /// assert_eq!(d.back(), None);
    ///
    /// d.push(1i);
    /// d.push(2i);
    /// match d.back_mut() {
    ///     Some(x) => *x = 9i,
    ///     None => (),
    /// }
    /// assert_eq!(d.back(), Some(&9i));
    /// ```
    fn back_mut<'a>(&'a mut self) -> Option<&'a mut T>;

    /// Inserts an element first in the sequence.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::{DList, Deque};
    ///
    /// let mut d = DList::new();
    /// d.push_front(1i);
    /// d.push_front(2i);
    /// assert_eq!(d.front(), Some(&2i));
    /// ```
    fn push_front(&mut self, elt: T);

    /// Inserts an element last in the sequence.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use std::collections::{DList, Deque};
    ///
    /// let mut d = DList::new();
    /// d.push_back(1i);
    /// d.push_back(2i);
    /// assert_eq!(d.front(), Some(&1i));
    /// ```
    #[deprecated = "use the `push` method"]
    fn push_back(&mut self, elt: T) { self.push(elt) }

    /// Removes the last element and returns it, or `None` if the sequence is
    /// empty.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use std::collections::{RingBuf, Deque};
    ///
    /// let mut d = RingBuf::new();
    /// d.push_back(1i);
    /// d.push_back(2i);
    ///
    /// assert_eq!(d.pop_back(), Some(2i));
    /// assert_eq!(d.pop_back(), Some(1i));
    /// assert_eq!(d.pop_back(), None);
    /// ```
    #[deprecated = "use the `pop` method"]
    fn pop_back(&mut self) -> Option<T> { self.pop() }

    /// Removes the first element and returns it, or `None` if the sequence is
    /// empty.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::{RingBuf, Deque};
    ///
    /// let mut d = RingBuf::new();
    /// d.push(1i);
    /// d.push(2i);
    ///
    /// assert_eq!(d.pop_front(), Some(1i));
    /// assert_eq!(d.pop_front(), Some(2i));
    /// assert_eq!(d.pop_front(), None);
    /// ```
    fn pop_front(&mut self) -> Option<T>;
}

// FIXME(#14344) this shouldn't be necessary
#[doc(hidden)]
pub fn fixme_14344_be_sure_to_link_to_collections() {}

#[cfg(not(test))]
mod std {
    pub use core::fmt;      // necessary for fail!()
    pub use core::option;   // necessary for fail!()
    pub use core::clone;    // deriving(Clone)
    pub use core::cmp;      // deriving(Eq, Ord, etc.)
    pub use hash;           // deriving(Hash)

    pub mod collections {
        pub use MutableSeq;
    }
}
