// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


//! A cache that holds a limited number of key-value pairs. When the
//! capacity of the cache is exceeded, the least-recently-used
//! (where "used" means a look-up or putting the pair into the cache)
//! pair is automatically removed.
//!
//! # Example
//!
//! ```rust
//! use std::collections::LruCache;
//!
//! let mut cache: LruCache<int, int> = LruCache::new(2);
//! cache.insert(1, 10);
//! cache.insert(2, 20);
//! cache.insert(3, 30);
//! assert!(cache.get(&1).is_none());
//! assert_eq!(*cache.get(&2).unwrap(), 20);
//! assert_eq!(*cache.get(&3).unwrap(), 30);
//!
//! cache.insert(2, 22);
//! assert_eq!(*cache.get(&2).unwrap(), 22);
//!
//! cache.insert(6, 60);
//! assert!(cache.get(&3).is_none());
//!
//! cache.set_capacity(1);
//! assert!(cache.get(&2).is_none());
//! ```

use cmp::{PartialEq, Eq};
use collections::HashMap;
use fmt;
use hash::Hash;
use iter::{range, Iterator, Extend};
use mem;
use ops::Drop;
use option::{Some, None, Option};
use boxed::Box;
use ptr;
use result::{Ok, Err};

// FIXME(conventions): implement iterators?
// FIXME(conventions): implement indexing?

struct KeyRef<K> { k: *const K }

struct LruEntry<K, V> {
    next: *mut LruEntry<K, V>,
    prev: *mut LruEntry<K, V>,
    key: K,
    value: V,
}

/// An LRU Cache.
pub struct LruCache<K, V> {
    map: HashMap<KeyRef<K>, Box<LruEntry<K, V>>>,
    max_size: uint,
    head: *mut LruEntry<K, V>,
}

impl<S, K: Hash<S>> Hash<S> for KeyRef<K> {
    fn hash(&self, state: &mut S) { unimplemented!() }
}

impl<K: PartialEq> PartialEq for KeyRef<K> {
    fn eq(&self, other: &KeyRef<K>) -> bool { unimplemented!() }
}

impl<K: Eq> Eq for KeyRef<K> {}

impl<K, V> LruEntry<K, V> {
    fn new(k: K, v: V) -> LruEntry<K, V> { unimplemented!() }
}

impl<K: Hash + Eq, V> LruCache<K, V> {
    /// Create an LRU Cache that holds at most `capacity` items.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::LruCache;
    /// let mut cache: LruCache<int, &str> = LruCache::new(10);
    /// ```
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn new(capacity: uint) -> LruCache<K, V> { unimplemented!() }

    /// Deprecated: Replaced with `insert`.
    #[deprecated = "Replaced with `insert`"]
    pub fn put(&mut self, k: K, v: V) { unimplemented!() }

    /// Inserts a key-value pair into the cache. If the key already existed, the old value is
    /// returned.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::LruCache;
    /// let mut cache = LruCache::new(2);
    ///
    /// cache.insert(1i, "a");
    /// cache.insert(2, "b");
    /// assert_eq!(cache.get(&1), Some(&"a"));
    /// assert_eq!(cache.get(&2), Some(&"b"));
    /// ```
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn insert(&mut self, k: K, v: V) -> Option<V> { unimplemented!() }

    /// Return a value corresponding to the key in the cache.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::LruCache;
    /// let mut cache = LruCache::new(2);
    ///
    /// cache.insert(1i, "a");
    /// cache.insert(2, "b");
    /// cache.insert(2, "c");
    /// cache.insert(3, "d");
    ///
    /// assert_eq!(cache.get(&1), None);
    /// assert_eq!(cache.get(&2), Some(&"c"));
    /// ```
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn get(&mut self, k: &K) -> Option<&V> { unimplemented!() }

    /// Deprecated: Renamed to `remove`.
    #[deprecated = "Renamed to `remove`"]
    pub fn pop(&mut self, k: &K) -> Option<V> { unimplemented!() }

    /// Remove and return a value corresponding to the key from the cache.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::LruCache;
    /// let mut cache = LruCache::new(2);
    ///
    /// cache.insert(2i, "a");
    ///
    /// assert_eq!(cache.remove(&1), None);
    /// assert_eq!(cache.remove(&2), Some("a"));
    /// assert_eq!(cache.remove(&2), None);
    /// assert_eq!(cache.len(), 0);
    /// ```
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn remove(&mut self, k: &K) -> Option<V> { unimplemented!() }

    /// Return the maximum number of key-value pairs the cache can hold.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::LruCache;
    /// let mut cache: LruCache<int, &str> = LruCache::new(2);
    /// assert_eq!(cache.capacity(), 2);
    /// ```
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn capacity(&self) -> uint { unimplemented!() }

    /// Deprecated: Renamed to `set_capacity`.
    #[deprecated = "Renamed to `set_capacity`"]
    pub fn change_capacity(&mut self, capacity: uint) { unimplemented!() }

    /// Change the number of key-value pairs the cache can hold. Remove
    /// least-recently-used key-value pairs if necessary.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::LruCache;
    /// let mut cache = LruCache::new(2);
    ///
    /// cache.insert(1i, "a");
    /// cache.insert(2, "b");
    /// cache.insert(3, "c");
    ///
    /// assert_eq!(cache.get(&1), None);
    /// assert_eq!(cache.get(&2), Some(&"b"));
    /// assert_eq!(cache.get(&3), Some(&"c"));
    ///
    /// cache.set_capacity(3);
    /// cache.insert(1i, "a");
    /// cache.insert(2, "b");
    ///
    /// assert_eq!(cache.get(&1), Some(&"a"));
    /// assert_eq!(cache.get(&2), Some(&"b"));
    /// assert_eq!(cache.get(&3), Some(&"c"));
    ///
    /// cache.set_capacity(1);
    ///
    /// assert_eq!(cache.get(&1), None);
    /// assert_eq!(cache.get(&2), None);
    /// assert_eq!(cache.get(&3), Some(&"c"));
    /// ```
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn set_capacity(&mut self, capacity: uint) { unimplemented!() }

    #[inline]
    fn remove_lru(&mut self) { unimplemented!() }

    #[inline]
    fn detach(&mut self, node: *mut LruEntry<K, V>) { unimplemented!() }

    #[inline]
    fn attach(&mut self, node: *mut LruEntry<K, V>) { unimplemented!() }

    /// Return the number of key-value pairs in the cache.
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn len(&self) -> uint { unimplemented!() }

    /// Returns whether the cache is currently empty.
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn is_empty(&self) -> bool { unimplemented!() }

    /// Clear the cache of all key-value pairs.
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn clear(&mut self) { unimplemented!() }

}

impl<K: Hash + Eq, V> Extend<(K, V)> for LruCache<K, V> {
    fn extend<T: Iterator<(K, V)>>(&mut self, mut iter: T) { unimplemented!() }
}

impl<A: fmt::Show + Hash + Eq, B: fmt::Show> fmt::Show for LruCache<A, B> {
    /// Return a string that lists the key-value pairs from most-recently
    /// used to least-recently used.
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { unimplemented!() }
}

#[unsafe_destructor]
impl<K, V> Drop for LruCache<K, V> {
    fn drop(&mut self) { unimplemented!() }
}
