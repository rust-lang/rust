// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! An ordered map and set for integer keys implemented as a radix trie

use prelude::*;
use iterator::{IteratorUtil, FromIterator};
use uint;
use util::{swap, replace};

// FIXME: #5244: need to manually update the TrieNode constructor
static SHIFT: uint = 4;
static SIZE: uint = 1 << SHIFT;
static MASK: uint = SIZE - 1;

enum Child<T> {
    Internal(~TrieNode<T>),
    External(uint, T),
    Nothing
}

#[allow(missing_doc)]
pub struct TrieMap<T> {
    priv root: TrieNode<T>,
    priv length: uint
}

impl<T> Container for TrieMap<T> {
    /// Return the number of elements in the map
    #[inline]
    fn len(&self) -> uint { self.length }

    /// Return true if the map contains no elements
    #[inline]
    fn is_empty(&self) -> bool { self.len() == 0 }
}

impl<T> Mutable for TrieMap<T> {
    /// Clear the map, removing all values.
    #[inline]
    fn clear(&mut self) {
        self.root = TrieNode::new();
        self.length = 0;
    }
}

impl<T> Map<uint, T> for TrieMap<T> {
    /// Return true if the map contains a value for the specified key
    #[inline]
    fn contains_key(&self, key: &uint) -> bool {
        self.find(key).is_some()
    }

    /// Return a reference to the value corresponding to the key
    #[inline]
    fn find<'a>(&'a self, key: &uint) -> Option<&'a T> {
        let mut node: &'a TrieNode<T> = &self.root;
        let mut idx = 0;
        loop {
            match node.children[chunk(*key, idx)] {
              Internal(ref x) => node = &**x,
              External(stored, ref value) => {
                if stored == *key {
                    return Some(value)
                } else {
                    return None
                }
              }
              Nothing => return None
            }
            idx += 1;
        }
    }
}

impl<T> MutableMap<uint, T> for TrieMap<T> {
    /// Return a mutable reference to the value corresponding to the key
    #[inline]
    fn find_mut<'a>(&'a mut self, key: &uint) -> Option<&'a mut T> {
        find_mut(&mut self.root.children[chunk(*key, 0)], *key, 1)
    }

    /// Insert a key-value pair into the map. An existing value for a
    /// key is replaced by the new value. Return true if the key did
    /// not already exist in the map.
    #[inline]
    fn insert(&mut self, key: uint, value: T) -> bool {
        self.swap(key, value).is_none()
    }

    /// Remove a key-value pair from the map. Return true if the key
    /// was present in the map, otherwise false.
    #[inline]
    fn remove(&mut self, key: &uint) -> bool {
        self.pop(key).is_some()
    }

    /// Insert a key-value pair from the map. If the key already had a value
    /// present in the map, that value is returned. Otherwise None is returned.
    fn swap(&mut self, key: uint, value: T) -> Option<T> {
        let ret = insert(&mut self.root.count,
                         &mut self.root.children[chunk(key, 0)],
                         key, value, 1);
        if ret.is_none() { self.length += 1 }
        ret
    }

    /// Removes a key from the map, returning the value at the key if the key
    /// was previously in the map.
    fn pop(&mut self, key: &uint) -> Option<T> {
        let ret = remove(&mut self.root.count,
                         &mut self.root.children[chunk(*key, 0)],
                         *key, 1);
        if ret.is_some() { self.length -= 1 }
        ret
    }
}

impl<T> TrieMap<T> {
    /// Create an empty TrieMap
    #[inline]
    pub fn new() -> TrieMap<T> {
        TrieMap{root: TrieNode::new(), length: 0}
    }

    /// Visit all key-value pairs in reverse order
    #[inline]
    pub fn each_reverse<'a>(&'a self, f: &fn(&uint, &'a T) -> bool) -> bool {
        self.root.each_reverse(f)
    }

    /// Visit all key-value pairs in order
    #[inline]
    pub fn each<'a>(&'a self, f: &fn(&uint, &'a T) -> bool) -> bool {
        self.root.each(f)
    }

    /// Visit all keys in order
    #[inline]
    pub fn each_key(&self, f: &fn(&uint) -> bool) -> bool {
        self.each(|k, _| f(k))
    }

    /// Visit all values in order
    #[inline]
    pub fn each_value<'a>(&'a self, f: &fn(&'a T) -> bool) -> bool {
        self.each(|_, v| f(v))
    }

    /// Iterate over the map and mutate the contained values
    #[inline]
    pub fn mutate_values(&mut self, f: &fn(&uint, &mut T) -> bool) -> bool {
        self.root.mutate_values(f)
    }

    /// Visit all keys in reverse order
    #[inline]
    pub fn each_key_reverse(&self, f: &fn(&uint) -> bool) -> bool {
        self.each_reverse(|k, _| f(k))
    }

    /// Visit all values in reverse order
    #[inline]
    pub fn each_value_reverse(&self, f: &fn(&T) -> bool) -> bool {
        self.each_reverse(|_, v| f(v))
    }
}

impl<T, Iter: Iterator<(uint, T)>> FromIterator<(uint, T), Iter> for TrieMap<T> {
    pub fn from_iterator(iter: &mut Iter) -> TrieMap<T> {
        let mut map = TrieMap::new();

        for iter.advance |(k, v)| {
            map.insert(k, v);
        }

        map
    }
}

#[allow(missing_doc)]
pub struct TrieSet {
    priv map: TrieMap<()>
}

impl Container for TrieSet {
    /// Return the number of elements in the set
    #[inline]
    fn len(&self) -> uint { self.map.len() }

    /// Return true if the set contains no elements
    #[inline]
    fn is_empty(&self) -> bool { self.map.is_empty() }
}

impl Mutable for TrieSet {
    /// Clear the set, removing all values.
    #[inline]
    fn clear(&mut self) { self.map.clear() }
}

impl TrieSet {
    /// Create an empty TrieSet
    #[inline]
    pub fn new() -> TrieSet {
        TrieSet{map: TrieMap::new()}
    }

    /// Return true if the set contains a value
    #[inline]
    pub fn contains(&self, value: &uint) -> bool {
        self.map.contains_key(value)
    }

    /// Add a value to the set. Return true if the value was not already
    /// present in the set.
    #[inline]
    pub fn insert(&mut self, value: uint) -> bool {
        self.map.insert(value, ())
    }

    /// Remove a value from the set. Return true if the value was
    /// present in the set.
    #[inline]
    pub fn remove(&mut self, value: &uint) -> bool {
        self.map.remove(value)
    }

    /// Visit all values in order
    #[inline]
    pub fn each(&self, f: &fn(&uint) -> bool) -> bool { self.map.each_key(f) }

    /// Visit all values in reverse order
    #[inline]
    pub fn each_reverse(&self, f: &fn(&uint) -> bool) -> bool {
        self.map.each_key_reverse(f)
    }
}

impl<Iter: Iterator<uint>> FromIterator<uint, Iter> for TrieSet {
    pub fn from_iterator(iter: &mut Iter) -> TrieSet {
        let mut set = TrieSet::new();

        for iter.advance |elem| {
            set.insert(elem);
        }

        set
    }
}

struct TrieNode<T> {
    count: uint,
    children: [Child<T>, ..SIZE]
}

impl<T> TrieNode<T> {
    #[inline]
    fn new() -> TrieNode<T> {
        // FIXME: #5244: [Nothing, ..SIZE] should be possible without implicit
        // copyability
        TrieNode{count: 0,
                 children: [Nothing, Nothing, Nothing, Nothing,
                            Nothing, Nothing, Nothing, Nothing,
                            Nothing, Nothing, Nothing, Nothing,
                            Nothing, Nothing, Nothing, Nothing]}
    }
}

impl<T> TrieNode<T> {
    fn each<'a>(&'a self, f: &fn(&uint, &'a T) -> bool) -> bool {
        for uint::range(0, self.children.len()) |idx| {
            match self.children[idx] {
                Internal(ref x) => if !x.each(|i,t| f(i,t)) { return false },
                External(k, ref v) => if !f(&k, v) { return false },
                Nothing => ()
            }
        }
        true
    }

    fn each_reverse<'a>(&'a self, f: &fn(&uint, &'a T) -> bool) -> bool {
        for uint::range_rev(self.children.len(), 0) |idx| {
            match self.children[idx] {
                Internal(ref x) => if !x.each_reverse(|i,t| f(i,t)) { return false },
                External(k, ref v) => if !f(&k, v) { return false },
                Nothing => ()
            }
        }
        true
    }

    fn mutate_values<'a>(&'a mut self, f: &fn(&uint, &mut T) -> bool) -> bool {
        for self.children.mut_iter().advance |child| {
            match *child {
                Internal(ref mut x) => if !x.mutate_values(|i,t| f(i,t)) {
                    return false
                },
                External(k, ref mut v) => if !f(&k, v) { return false },
                Nothing => ()
            }
        }
        true
    }
}

// if this was done via a trait, the key could be generic
#[inline]
fn chunk(n: uint, idx: uint) -> uint {
    let sh = uint::bits - (SHIFT * (idx + 1));
    (n >> sh) & MASK
}

fn find_mut<'r, T>(child: &'r mut Child<T>, key: uint, idx: uint) -> Option<&'r mut T> {
    match *child {
        External(_, ref mut value) => Some(value),
        Internal(ref mut x) => find_mut(&mut x.children[chunk(key, idx)], key, idx + 1),
        Nothing => None
    }
}

fn insert<T>(count: &mut uint, child: &mut Child<T>, key: uint, value: T,
             idx: uint) -> Option<T> {
    let mut tmp = Nothing;
    let ret;
    swap(&mut tmp, child);

    *child = match tmp {
      External(stored_key, stored_value) => {
          if stored_key == key {
              ret = Some(stored_value);
              External(stored_key, value)
          } else {
              // conflict - split the node
              let mut new = ~TrieNode::new();
              insert(&mut new.count,
                     &mut new.children[chunk(stored_key, idx)],
                     stored_key, stored_value, idx + 1);
              ret = insert(&mut new.count, &mut new.children[chunk(key, idx)],
                           key, value, idx + 1);
              Internal(new)
          }
      }
      Internal(x) => {
        let mut x = x;
        ret = insert(&mut x.count, &mut x.children[chunk(key, idx)], key,
                     value, idx + 1);
        Internal(x)
      }
      Nothing => {
        *count += 1;
        ret = None;
        External(key, value)
      }
    };
    return ret;
}

fn remove<T>(count: &mut uint, child: &mut Child<T>, key: uint,
             idx: uint) -> Option<T> {
    let (ret, this) = match *child {
      External(stored, _) if stored == key => {
        match replace(child, Nothing) {
            External(_, value) => (Some(value), true),
            _ => fail!()
        }
      }
      External(*) => (None, false),
      Internal(ref mut x) => {
          let ret = remove(&mut x.count, &mut x.children[chunk(key, idx)],
                           key, idx + 1);
          (ret, x.count == 0)
      }
      Nothing => (None, false)
    };

    if this {
        *child = Nothing;
        *count -= 1;
    }
    return ret;
}

#[cfg(test)]
pub fn check_integrity<T>(trie: &TrieNode<T>) {
    assert!(trie.count != 0);

    let mut sum = 0;

    for trie.children.iter().advance |x| {
        match *x {
          Nothing => (),
          Internal(ref y) => {
              check_integrity(&**y);
              sum += 1
          }
          External(_, _) => { sum += 1 }
        }
    }

    assert_eq!(sum, trie.count);
}

#[cfg(test)]
mod test_map {
    use super::*;
    use core::option::{Some, None};
    use uint;

    #[test]
    fn test_find_mut() {
        let mut m = TrieMap::new();
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
    fn test_step() {
        let mut trie = TrieMap::new();
        let n = 300;

        for uint::range_step(1, n, 2) |x| {
            assert!(trie.insert(x, x + 1));
            assert!(trie.contains_key(&x));
            check_integrity(&trie.root);
        }

        for uint::range_step(0, n, 2) |x| {
            assert!(!trie.contains_key(&x));
            assert!(trie.insert(x, x + 1));
            check_integrity(&trie.root);
        }

        for uint::range(0, n) |x| {
            assert!(trie.contains_key(&x));
            assert!(!trie.insert(x, x + 1));
            check_integrity(&trie.root);
        }

        for uint::range_step(1, n, 2) |x| {
            assert!(trie.remove(&x));
            assert!(!trie.contains_key(&x));
            check_integrity(&trie.root);
        }

        for uint::range_step(0, n, 2) |x| {
            assert!(trie.contains_key(&x));
            assert!(!trie.insert(x, x + 1));
            check_integrity(&trie.root);
        }
    }

    #[test]
    fn test_each() {
        let mut m = TrieMap::new();

        assert!(m.insert(3, 6));
        assert!(m.insert(0, 0));
        assert!(m.insert(4, 8));
        assert!(m.insert(2, 4));
        assert!(m.insert(1, 2));

        let mut n = 0;
        for m.each |k, v| {
            assert_eq!(*k, n);
            assert_eq!(*v, n * 2);
            n += 1;
        }
    }

    #[test]
    fn test_each_break() {
        let mut m = TrieMap::new();

        for uint::range_rev(uint::max_value, uint::max_value - 10000) |x| {
            m.insert(x, x / 2);
        }

        let mut n = uint::max_value - 10000;
        for m.each |k, v| {
            if n == uint::max_value - 5000 { break }
            assert!(n < uint::max_value - 5000);

            assert_eq!(*k, n);
            assert_eq!(*v, n / 2);
            n += 1;
        }
    }

    #[test]
    fn test_each_reverse() {
        let mut m = TrieMap::new();

        assert!(m.insert(3, 6));
        assert!(m.insert(0, 0));
        assert!(m.insert(4, 8));
        assert!(m.insert(2, 4));
        assert!(m.insert(1, 2));

        let mut n = 4;
        for m.each_reverse |k, v| {
            assert_eq!(*k, n);
            assert_eq!(*v, n * 2);
            n -= 1;
        }
    }

    #[test]
    fn test_each_reverse_break() {
        let mut m = TrieMap::new();

        for uint::range_rev(uint::max_value, uint::max_value - 10000) |x| {
            m.insert(x, x / 2);
        }

        let mut n = uint::max_value - 1;
        for m.each_reverse |k, v| {
            if n == uint::max_value - 5000 { break }
            assert!(n > uint::max_value - 5000);

            assert_eq!(*k, n);
            assert_eq!(*v, n / 2);
            n -= 1;
        }
    }

    #[test]
    fn test_swap() {
        let mut m = TrieMap::new();
        assert_eq!(m.swap(1, 2), None);
        assert_eq!(m.swap(1, 3), Some(2));
        assert_eq!(m.swap(1, 4), Some(3));
    }

    #[test]
    fn test_pop() {
        let mut m = TrieMap::new();
        m.insert(1, 2);
        assert_eq!(m.pop(&1), Some(2));
        assert_eq!(m.pop(&1), None);
    }

    #[test]
    fn test_from_iter() {
        let xs = ~[(1u, 1i), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6)];

        let map: TrieMap<int> = xs.iter().transform(|&x| x).collect();

        for xs.iter().advance |&(k, v)| {
            assert_eq!(map.find(&k), Some(&v));
        }
    }
}

#[cfg(test)]
mod test_set {
    use super::*;
    use uint;

    #[test]
    fn test_sane_chunk() {
        let x = 1;
        let y = 1 << (uint::bits - 1);

        let mut trie = TrieSet::new();

        assert!(trie.insert(x));
        assert!(trie.insert(y));

        assert_eq!(trie.len(), 2);

        let expected = [x, y];

        let mut i = 0;

        for trie.each |x| {
            assert_eq!(expected[i], *x);
            i += 1;
        }
    }

    #[test]
    fn test_from_iter() {
        let xs = ~[9u, 8, 7, 6, 5, 4, 3, 2, 1];

        let set: TrieSet = xs.iter().transform(|&x| x).collect();

        for xs.iter().advance |x| {
            assert!(set.contains(x));
        }
    }
}
