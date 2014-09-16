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
//! cache.put(1, 10);
//! cache.put(2, 20);
//! cache.put(3, 30);
//! assert!(cache.get(&1).is_none());
//! assert_eq!(*cache.get(&2).unwrap(), 20);
//! assert_eq!(*cache.get(&3).unwrap(), 30);
//!
//! cache.put(2, 22);
//! assert_eq!(*cache.get(&2).unwrap(), 22);
//!
//! cache.put(6, 60);
//! assert!(cache.get(&3).is_none());
//!
//! cache.change_capacity(1);
//! assert!(cache.get(&2).is_none());
//! ```

use cmp::{PartialEq, Eq};
use collections::{HashMap, Collection, Mutable, MutableMap};
use fmt;
use hash::Hash;
use iter::{range, Iterator};
use mem;
use ops::Drop;
use option::{Some, None, Option};
use boxed::Box;
use ptr;
use result::{Ok, Err};

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
    fn hash(&self, state: &mut S) {
        unsafe { (*self.k).hash(state) }
    }
}

impl<K: PartialEq> PartialEq for KeyRef<K> {
    fn eq(&self, other: &KeyRef<K>) -> bool {
        unsafe{ (*self.k).eq(&*other.k) }
    }
}

impl<K: Eq> Eq for KeyRef<K> {}

impl<K, V> LruEntry<K, V> {
    fn new(k: K, v: V) -> LruEntry<K, V> {
        LruEntry {
            key: k,
            value: v,
            next: ptr::null_mut(),
            prev: ptr::null_mut(),
        }
    }
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
    pub fn new(capacity: uint) -> LruCache<K, V> {
        let cache = LruCache {
            map: HashMap::new(),
            max_size: capacity,
            head: unsafe{ mem::transmute(box mem::uninitialized::<LruEntry<K, V>>()) },
        };
        unsafe {
            (*cache.head).next = cache.head;
            (*cache.head).prev = cache.head;
        }
        return cache;
    }

    /// Put a key-value pair into cache.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::LruCache;
    /// let mut cache = LruCache::new(2);
    ///
    /// cache.put(1i, "a");
    /// cache.put(2, "b");
    /// assert_eq!(cache.get(&1), Some(&"a"));
    /// assert_eq!(cache.get(&2), Some(&"b"));
    /// ```
    pub fn put(&mut self, k: K, v: V) {
        let (node_ptr, node_opt) = match self.map.find_mut(&KeyRef{k: &k}) {
            Some(node) => {
                node.value = v;
                let node_ptr: *mut LruEntry<K, V> = &mut **node;
                (node_ptr, None)
            }
            None => {
                let mut node = box LruEntry::new(k, v);
                let node_ptr: *mut LruEntry<K, V> = &mut *node;
                (node_ptr, Some(node))
            }
        };
        match node_opt {
            None => {
                // Existing node, just update LRU position
                self.detach(node_ptr);
                self.attach(node_ptr);
            }
            Some(node) => {
                let keyref = unsafe { &(*node_ptr).key };
                self.map.swap(KeyRef{k: keyref}, node);
                self.attach(node_ptr);
                if self.len() > self.capacity() {
                    self.remove_lru();
                }
            }
        }
    }

    /// Return a value corresponding to the key in the cache.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::LruCache;
    /// let mut cache = LruCache::new(2);
    ///
    /// cache.put(1i, "a");
    /// cache.put(2, "b");
    /// cache.put(2, "c");
    /// cache.put(3, "d");
    ///
    /// assert_eq!(cache.get(&1), None);
    /// assert_eq!(cache.get(&2), Some(&"c"));
    /// ```
    pub fn get<'a>(&'a mut self, k: &K) -> Option<&'a V> {
        let (value, node_ptr_opt) = match self.map.find_mut(&KeyRef{k: k}) {
            None => (None, None),
            Some(node) => {
                let node_ptr: *mut LruEntry<K, V> = &mut **node;
                (Some(unsafe { &(*node_ptr).value }), Some(node_ptr))
            }
        };
        match node_ptr_opt {
            None => (),
            Some(node_ptr) => {
                self.detach(node_ptr);
                self.attach(node_ptr);
            }
        }
        return value;
    }

    /// Remove and return a value corresponding to the key from the cache.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::LruCache;
    /// let mut cache = LruCache::new(2);
    ///
    /// cache.put(2i, "a");
    ///
    /// assert_eq!(cache.pop(&1), None);
    /// assert_eq!(cache.pop(&2), Some("a"));
    /// assert_eq!(cache.pop(&2), None);
    /// assert_eq!(cache.len(), 0);
    /// ```
    pub fn pop(&mut self, k: &K) -> Option<V> {
        match self.map.pop(&KeyRef{k: k}) {
            None => None,
            Some(lru_entry) => Some(lru_entry.value)
        }
    }

    /// Return the maximum number of key-value pairs the cache can hold.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::LruCache;
    /// let mut cache: LruCache<int, &str> = LruCache::new(2);
    /// assert_eq!(cache.capacity(), 2);
    /// ```
    pub fn capacity(&self) -> uint {
        self.max_size
    }

    /// Change the number of key-value pairs the cache can hold. Remove
    /// least-recently-used key-value pairs if necessary.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::LruCache;
    /// let mut cache = LruCache::new(2);
    ///
    /// cache.put(1i, "a");
    /// cache.put(2, "b");
    /// cache.put(3, "c");
    ///
    /// assert_eq!(cache.get(&1), None);
    /// assert_eq!(cache.get(&2), Some(&"b"));
    /// assert_eq!(cache.get(&3), Some(&"c"));
    ///
    /// cache.change_capacity(3);
    /// cache.put(1i, "a");
    /// cache.put(2, "b");
    ///
    /// assert_eq!(cache.get(&1), Some(&"a"));
    /// assert_eq!(cache.get(&2), Some(&"b"));
    /// assert_eq!(cache.get(&3), Some(&"c"));
    ///
    /// cache.change_capacity(1);
    ///
    /// assert_eq!(cache.get(&1), None);
    /// assert_eq!(cache.get(&2), None);
    /// assert_eq!(cache.get(&3), Some(&"c"));
    /// ```
    pub fn change_capacity(&mut self, capacity: uint) {
        for _ in range(capacity, self.len()) {
            self.remove_lru();
        }
        self.max_size = capacity;
    }

    #[inline]
    fn remove_lru(&mut self) {
        if self.len() > 0 {
            let lru = unsafe { (*self.head).prev };
            self.detach(lru);
            self.map.pop(&KeyRef{k: unsafe { &(*lru).key }});
        }
    }

    #[inline]
    fn detach(&mut self, node: *mut LruEntry<K, V>) {
        unsafe {
            (*(*node).prev).next = (*node).next;
            (*(*node).next).prev = (*node).prev;
        }
    }

    #[inline]
    fn attach(&mut self, node: *mut LruEntry<K, V>) {
        unsafe {
            (*node).next = (*self.head).next;
            (*node).prev = self.head;
            (*self.head).next = node;
            (*(*node).next).prev = node;
        }
    }
}

impl<A: fmt::Show + Hash + Eq, B: fmt::Show> fmt::Show for LruCache<A, B> {
    /// Return a string that lists the key-value pairs from most-recently
    /// used to least-recently used.
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        try!(write!(f, "{{"));
        let mut cur = self.head;
        for i in range(0, self.len()) {
            if i > 0 { try!(write!(f, ", ")) }
            unsafe {
                cur = (*cur).next;
                try!(write!(f, "{}", (*cur).key));
            }
            try!(write!(f, ": "));
            unsafe {
                try!(write!(f, "{}", (*cur).value));
            }
        }
        write!(f, r"}}")
    }
}

impl<K: Hash + Eq, V> Collection for LruCache<K, V> {
    /// Return the number of key-value pairs in the cache.
    fn len(&self) -> uint {
        self.map.len()
    }
}

impl<K: Hash + Eq, V> Mutable for LruCache<K, V> {
    /// Clear the cache of all key-value pairs.
    fn clear(&mut self) {
        self.map.clear();
    }
}

#[unsafe_destructor]
impl<K, V> Drop for LruCache<K, V> {
    fn drop(&mut self) {
        unsafe {
            let node: Box<LruEntry<K, V>> = mem::transmute(self.head);
            // Prevent compiler from trying to drop the un-initialized field in the sigil node.
            let box internal_node = node;
            let LruEntry { next: _, prev: _, key: k, value: v } = internal_node;
            mem::forget(k);
            mem::forget(v);
        }
    }
}

#[cfg(test)]
mod tests {
    use prelude::*;
    use super::LruCache;

    fn assert_opt_eq<V: PartialEq>(opt: Option<&V>, v: V) {
        assert!(opt.is_some());
        assert!(opt.unwrap() == &v);
    }

    #[test]
    fn test_put_and_get() {
        let mut cache: LruCache<int, int> = LruCache::new(2);
        cache.put(1, 10);
        cache.put(2, 20);
        assert_opt_eq(cache.get(&1), 10);
        assert_opt_eq(cache.get(&2), 20);
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn test_put_update() {
        let mut cache: LruCache<String, Vec<u8>> = LruCache::new(1);
        cache.put("1".to_string(), vec![10, 10]);
        cache.put("1".to_string(), vec![10, 19]);
        assert_opt_eq(cache.get(&"1".to_string()), vec![10, 19]);
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_expire_lru() {
        let mut cache: LruCache<String, String> = LruCache::new(2);
        cache.put("foo1".to_string(), "bar1".to_string());
        cache.put("foo2".to_string(), "bar2".to_string());
        cache.put("foo3".to_string(), "bar3".to_string());
        assert!(cache.get(&"foo1".to_string()).is_none());
        cache.put("foo2".to_string(), "bar2update".to_string());
        cache.put("foo4".to_string(), "bar4".to_string());
        assert!(cache.get(&"foo3".to_string()).is_none());
    }

    #[test]
    fn test_pop() {
        let mut cache: LruCache<int, int> = LruCache::new(2);
        cache.put(1, 10);
        cache.put(2, 20);
        assert_eq!(cache.len(), 2);
        let opt1 = cache.pop(&1);
        assert!(opt1.is_some());
        assert_eq!(opt1.unwrap(), 10);
        assert!(cache.get(&1).is_none());
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_change_capacity() {
        let mut cache: LruCache<int, int> = LruCache::new(2);
        assert_eq!(cache.capacity(), 2);
        cache.put(1, 10);
        cache.put(2, 20);
        cache.change_capacity(1);
        assert!(cache.get(&1).is_none());
        assert_eq!(cache.capacity(), 1);
    }

    #[test]
    fn test_to_string() {
        let mut cache: LruCache<int, int> = LruCache::new(3);
        cache.put(1, 10);
        cache.put(2, 20);
        cache.put(3, 30);
        assert_eq!(cache.to_string(), "{3: 30, 2: 20, 1: 10}".to_string());
        cache.put(2, 22);
        assert_eq!(cache.to_string(), "{2: 22, 3: 30, 1: 10}".to_string());
        cache.put(6, 60);
        assert_eq!(cache.to_string(), "{6: 60, 2: 22, 3: 30}".to_string());
        cache.get(&3);
        assert_eq!(cache.to_string(), "{3: 30, 6: 60, 2: 22}".to_string());
        cache.change_capacity(2);
        assert_eq!(cache.to_string(), "{3: 30, 6: 60}".to_string());
    }

    #[test]
    fn test_clear() {
        let mut cache: LruCache<int, int> = LruCache::new(2);
        cache.put(1, 10);
        cache.put(2, 20);
        cache.clear();
        assert!(cache.get(&1).is_none());
        assert!(cache.get(&2).is_none());
        assert_eq!(cache.to_string(), "{}".to_string());
    }
}
