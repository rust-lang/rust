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
//! use extra::lru_cache::LruCache;
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

use std::container::Container;
use std::hashmap::HashMap;
use std::to_bytes::Cb;
use std::ptr;
use std::cast;

struct KeyRef<K> { priv k: *K }

struct LruEntry<K, V> {
    priv key: Option<K>,
    priv value: Option<V>,
    priv next: *mut LruEntry<K, V>,
    priv prev: *mut LruEntry<K, V>,
}

/// An LRU Cache.
pub struct LruCache<K, V> {
    priv map: HashMap<KeyRef<K>, ~LruEntry<K, V>>,
    priv max_size: uint,
    priv head: *mut LruEntry<K, V>,
    priv tail: *mut LruEntry<K, V>,
}

impl<K: IterBytes> IterBytes for KeyRef<K> {
    fn iter_bytes(&self, lsb0: bool, f: Cb) -> bool {
        unsafe{ (*self.k).iter_bytes(lsb0, f) }
    }
}

impl<K: Eq> Eq for KeyRef<K> {
    fn eq(&self, other: &KeyRef<K>) -> bool {
        unsafe{ (*self.k).eq(&*other.k) }
    }
}

impl<K, V> LruEntry<K, V> {
    fn new() -> LruEntry<K, V> {
        LruEntry {
            key: None,
            value: None,
            next: ptr::mut_null(),
            prev: ptr::mut_null(),
        }
    }

    fn with_key_value(k: K, v: V) -> LruEntry<K, V> {
        LruEntry {
            key: Some(k),
            value: Some(v),
            next: ptr::mut_null(),
            prev: ptr::mut_null(),
        }
    }
}

impl<K: IterBytes + Eq, V> LruCache<K, V> {
    /// Create an LRU Cache that holds at most `capacity` items.
    pub fn new(capacity: uint) -> LruCache<K, V> {
        let cache = LruCache {
            map: HashMap::new(),
            max_size: capacity,
            head: unsafe{ cast::transmute(~LruEntry::<K, V>::new()) },
            tail: unsafe{ cast::transmute(~LruEntry::<K, V>::new()) },
        };
        unsafe {
            (*cache.head).next = cache.tail;
            (*cache.tail).prev = cache.head;
        }
        return cache;
    }

    /// Put a key-value pair into cache.
    pub fn put(&mut self, k: K, v: V) {
        let mut key_existed = false;
        let (node_ptr, node_opt) = match self.map.find_mut(&KeyRef{k: &k}) {
            Some(node) => {
                key_existed = true;
                node.value = Some(v);
                let node_ptr: *mut LruEntry<K, V> = &mut **node;
                (node_ptr, None)
            }
            None => {
                let mut node = ~LruEntry::with_key_value(k, v);
                let node_ptr: *mut LruEntry<K, V> = &mut *node;
                (node_ptr, Some(node))
            }
        };
        if key_existed {
            self.detach(node_ptr);
            self.attach(node_ptr);
        } else {
            let keyref = unsafe { (*node_ptr).key.as_ref().unwrap() };
            self.map.swap(KeyRef{k: keyref}, node_opt.unwrap());
            self.attach(node_ptr);
            if self.len() > self.capacity() {
                self.remove_lru();
            }
        }
    }

    /// Return a value corresponding to the key in the cache.
    pub fn get<'a>(&'a mut self, k: &K) -> Option<&'a V> {
        let (value, node_ptr_opt) = match self.map.find_mut(&KeyRef{k: k}) {
            None => (None, None),
            Some(node) => {
                let node_ptr: *mut LruEntry<K, V> = &mut **node;
                unsafe {
                    match (*node_ptr).value {
                        None => (None, None),
                        Some(ref value) => (Some(value), Some(node_ptr))
                    }
                }
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
    pub fn pop(&mut self, k: &K) -> Option<V> {
        match self.map.pop(&KeyRef{k: k}) {
            None => None,
            Some(lru_entry) => lru_entry.value
        }
    }

    /// Return the maximum number of key-value pairs the cache can hold.
    pub fn capacity(&self) -> uint {
        self.max_size
    }

    /// Change the number of key-value pairs the cache can hold. Remove
    /// least-recently-used key-value pairs if necessary.
    pub fn change_capacity(&mut self, capacity: uint) {
        for _ in range(capacity, self.len()) {
            self.remove_lru();
        }
        self.max_size = capacity;
    }

    #[inline]
    fn remove_lru(&mut self) {
        if self.len() > 0 {
            let lru = unsafe { (*self.tail).prev };
            self.detach(lru);
            unsafe {
                match (*lru).key {
                    None => (),
                    Some(ref k) => { self.map.pop(&KeyRef{k: k}); }
                }
            }
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

impl<A: ToStr + IterBytes + Eq, B: ToStr> ToStr for LruCache<A, B> {
    /// Return a string that lists the key-value pairs from most-recently
    /// used to least-recently used.
    #[inline]
    fn to_str(&self) -> ~str {
        let mut acc = ~"{";
        let mut cur = self.head;
        for i in range(0, self.len()) {
            if i > 0 {
                acc.push_str(", ");
            }
            unsafe {
                cur = (*cur).next;
                match (*cur).key {
                    // should never print nil
                    None => acc.push_str("nil"),
                    Some(ref k) => acc.push_str(k.to_str())
                }
            }
            acc.push_str(": ");
            unsafe {
                match (*cur).value {
                    // should never print nil
                    None => acc.push_str("nil"),
                    Some(ref value) => acc.push_str(value.to_str())
                }
            }
        }
        acc.push_char('}');
        acc
    }
}

impl<K: IterBytes + Eq, V> Container for LruCache<K, V> {
    /// Return the number of key-value pairs in the cache.
    fn len(&self) -> uint {
        self.map.len()
    }
}

impl<K: IterBytes + Eq, V> Mutable for LruCache<K, V> {
    /// Clear the cache of all key-value pairs.
    fn clear(&mut self) {
        self.map.clear();
    }
}

#[unsafe_destructor]
impl<K, V> Drop for LruCache<K, V> {
    fn drop(&mut self) {
        unsafe {
            let _: ~LruEntry<K, V> = cast::transmute(self.head);
            let _: ~LruEntry<K, V> = cast::transmute(self.tail);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::LruCache;

    fn assert_opt_eq<V: Eq>(opt: Option<&V>, v: V) {
        assert!(opt.is_some());
        assert_eq!(opt.unwrap(), &v);
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
        let mut cache: LruCache<~str, ~[u8]> = LruCache::new(1);
        cache.put(~"1", ~[10, 10]);
        cache.put(~"1", ~[10, 19]);
        assert_opt_eq(cache.get(&~"1"), ~[10, 19]);
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_expire_lru() {
        let mut cache: LruCache<~str, ~str> = LruCache::new(2);
        cache.put(~"foo1", ~"bar1");
        cache.put(~"foo2", ~"bar2");
        cache.put(~"foo3", ~"bar3");
        assert!(cache.get(&~"foo1").is_none());
        cache.put(~"foo2", ~"bar2update");
        cache.put(~"foo4", ~"bar4");
        assert!(cache.get(&~"foo3").is_none());
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
    fn test_to_str() {
        let mut cache: LruCache<int, int> = LruCache::new(3);
        cache.put(1, 10);
        cache.put(2, 20);
        cache.put(3, 30);
        assert_eq!(cache.to_str(), ~"{3: 30, 2: 20, 1: 10}");
        cache.put(2, 22);
        assert_eq!(cache.to_str(), ~"{2: 22, 3: 30, 1: 10}");
        cache.put(6, 60);
        assert_eq!(cache.to_str(), ~"{6: 60, 2: 22, 3: 30}");
        cache.get(&3);
        assert_eq!(cache.to_str(), ~"{3: 30, 6: 60, 2: 22}");
        cache.change_capacity(2);
        assert_eq!(cache.to_str(), ~"{3: 30, 6: 60}");
    }

    #[test]
    fn test_clear() {
        let mut cache: LruCache<int, int> = LruCache::new(2);
        cache.put(1, 10);
        cache.put(2, 20);
        cache.clear();
        assert!(cache.get(&1).is_none());
        assert!(cache.get(&2).is_none());
        assert_eq!(cache.to_str(), ~"{}");
    }
}
