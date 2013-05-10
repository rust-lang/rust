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

pub struct TrieMap<T> {
    priv root: TrieNode<T>,
    priv length: uint
}

impl<T> Container for TrieMap<T> {
    /// Return the number of elements in the map
    #[inline(always)]
    fn len(&const self) -> uint { self.length }

    /// Return true if the map contains no elements
    #[inline(always)]
    fn is_empty(&const self) -> bool { self.len() == 0 }
}

impl<T> Mutable for TrieMap<T> {
    /// Clear the map, removing all values.
    #[inline(always)]
    fn clear(&mut self) {
        self.root = TrieNode::new();
        self.length = 0;
    }
}

impl<T> Map<uint, T> for TrieMap<T> {
    /// Return true if the map contains a value for the specified key
    #[inline(always)]
    fn contains_key(&self, key: &uint) -> bool {
        self.find(key).is_some()
    }

    /// Visit all key-value pairs in order
    #[inline(always)]
    fn each<'a>(&'a self, f: &fn(&uint, &'a T) -> bool) {
        self.root.each(f);
    }

    /// Visit all keys in order
    #[inline(always)]
    fn each_key(&self, f: &fn(&uint) -> bool) {
        self.each(|k, _| f(k))
    }

    /// Visit all values in order
    #[inline(always)]
    fn each_value<'a>(&'a self, f: &fn(&'a T) -> bool) {
        self.each(|_, v| f(v))
    }

    /// Iterate over the map and mutate the contained values
    #[inline(always)]
    fn mutate_values(&mut self, f: &fn(&uint, &mut T) -> bool) {
        self.root.mutate_values(f);
    }

    /// Return a reference to the value corresponding to the key
    #[inline(hint)]
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

    /// Return a mutable reference to the value corresponding to the key
    #[inline(always)]
    fn find_mut<'a>(&'a mut self, key: &uint) -> Option<&'a mut T> {
        find_mut(&mut self.root.children[chunk(*key, 0)], *key, 1)
    }

    /// Insert a key-value pair into the map. An existing value for a
    /// key is replaced by the new value. Return true if the key did
    /// not already exist in the map.
    #[inline(always)]
    fn insert(&mut self, key: uint, value: T) -> bool {
        self.swap(key, value).is_none()
    }

    /// Remove a key-value pair from the map. Return true if the key
    /// was present in the map, otherwise false.
    #[inline(always)]
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

pub impl<T> TrieMap<T> {
    /// Create an empty TrieMap
    #[inline(always)]
    fn new() -> TrieMap<T> {
        TrieMap{root: TrieNode::new(), length: 0}
    }

    /// Visit all key-value pairs in reverse order
    #[inline(always)]
    fn each_reverse<'a>(&'a self, f: &fn(&uint, &'a T) -> bool) {
        self.root.each_reverse(f);
    }

    /// Visit all keys in reverse order
    #[inline(always)]
    fn each_key_reverse(&self, f: &fn(&uint) -> bool) {
        self.each_reverse(|k, _| f(k))
    }

    /// Visit all values in reverse order
    #[inline(always)]
    fn each_value_reverse(&self, f: &fn(&T) -> bool) {
        self.each_reverse(|_, v| f(v))
    }
}

pub struct TrieSet {
    priv map: TrieMap<()>
}

impl BaseIter<uint> for TrieSet {
    /// Visit all values in order
    #[inline(always)]
    fn each(&self, f: &fn(&uint) -> bool) { self.map.each_key(f) }
    #[inline(always)]
    fn size_hint(&self) -> Option<uint> { Some(self.len()) }
}

impl ReverseIter<uint> for TrieSet {
    /// Visit all values in reverse order
    #[inline(always)]
    fn each_reverse(&self, f: &fn(&uint) -> bool) {
        self.map.each_key_reverse(f)
    }
}

impl Container for TrieSet {
    /// Return the number of elements in the set
    #[inline(always)]
    fn len(&const self) -> uint { self.map.len() }

    /// Return true if the set contains no elements
    #[inline(always)]
    fn is_empty(&const self) -> bool { self.map.is_empty() }
}

impl Mutable for TrieSet {
    /// Clear the set, removing all values.
    #[inline(always)]
    fn clear(&mut self) { self.map.clear() }
}

pub impl TrieSet {
    /// Create an empty TrieSet
    #[inline(always)]
    fn new() -> TrieSet {
        TrieSet{map: TrieMap::new()}
    }

    /// Return true if the set contains a value
    #[inline(always)]
    fn contains(&self, value: &uint) -> bool {
        self.map.contains_key(value)
    }

    /// Add a value to the set. Return true if the value was not already
    /// present in the set.
    #[inline(always)]
    fn insert(&mut self, value: uint) -> bool { self.map.insert(value, ()) }

    /// Remove a value from the set. Return true if the value was
    /// present in the set.
    #[inline(always)]
    fn remove(&mut self, value: &uint) -> bool { self.map.remove(value) }
}

struct TrieNode<T> {
    count: uint,
    children: [Child<T>, ..SIZE]
}

impl<T> TrieNode<T> {
    #[inline(always)]
    fn new() -> TrieNode<T> {
        // FIXME: #5244: [Nothing, ..SIZE] should be possible without Copy
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
                Internal(ref x) => if !x.each(f) { return false },
                External(k, ref v) => if !f(&k, v) { return false },
                Nothing => ()
            }
        }
        true
    }

    fn each_reverse<'a>(&'a self, f: &fn(&uint, &'a T) -> bool) -> bool {
        for uint::range_rev(self.children.len(), 0) |idx| {
            match self.children[idx - 1] {
                Internal(ref x) => if !x.each_reverse(f) { return false },
                External(k, ref v) => if !f(&k, v) { return false },
                Nothing => ()
            }
        }
        true
    }

    fn mutate_values<'a>(&'a mut self, f: &fn(&uint, &mut T) -> bool) -> bool {
        for vec::each_mut(self.children) |child| {
            match *child {
                Internal(ref mut x) => if !x.mutate_values(f) {
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
#[inline(always)]
fn chunk(n: uint, idx: uint) -> uint {
    let sh = uint::bits - (SHIFT * (idx + 1));
    (n >> sh) & MASK
}

#[cfg(stage0)]
fn find_mut<'r, T>(child: &'r mut Child<T>, key: uint, idx: uint) -> Option<&'r mut T> {
    unsafe {
        (match *child {
            External(_, ref value) => Some(cast::transmute_mut(value)),
            Internal(ref x) => find_mut(cast::transmute_mut(&x.children[chunk(key, idx)]),
                                        key, idx + 1),
            Nothing => None
        }).map_consume(|x| cast::transmute_mut_region(x))
    }
}

#[cfg(not(stage0))]
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

    for trie.children.each |x| {
        match *x {
          Nothing => (),
          Internal(ref y) => {
              check_integrity(&**y);
              sum += 1
          }
          External(_, _) => { sum += 1 }
        }
    }

    assert!(sum == trie.count);
}

#[cfg(test)]
mod tests {
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
            assert!(*k == n);
            assert!(*v == n * 2);
            n += 1;
        }
    }

    #[test]
    fn test_each_break() {
        let mut m = TrieMap::new();

        for uint::range_rev(uint::max_value, uint::max_value - 10000) |x| {
            m.insert(x, x / 2);
        }

        let mut n = uint::max_value - 9999;
        for m.each |k, v| {
            if n == uint::max_value - 5000 { break }
            assert!(n < uint::max_value - 5000);

            assert!(*k == n);
            assert!(*v == n / 2);
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
            assert!(*k == n);
            assert!(*v == n * 2);
            n -= 1;
        }
    }

    #[test]
    fn test_each_reverse_break() {
        let mut m = TrieMap::new();

        for uint::range_rev(uint::max_value, uint::max_value - 10000) |x| {
            m.insert(x, x / 2);
        }

        let mut n = uint::max_value;
        for m.each_reverse |k, v| {
            if n == uint::max_value - 5000 { break }
            assert!(n > uint::max_value - 5000);

            assert!(*k == n);
            assert!(*v == n / 2);
            n -= 1;
        }
    }

    #[test]
    fn test_sane_chunk() {
        let x = 1;
        let y = 1 << (uint::bits - 1);

        let mut trie = TrieSet::new();

        assert!(trie.insert(x));
        assert!(trie.insert(y));

        assert!(trie.len() == 2);

        let expected = [x, y];

        let mut i = 0;

        for trie.each |x| {
            assert!(expected[i] == *x);
            i += 1;
        }
    }

    #[test]
    fn test_swap() {
        let mut m = TrieMap::new();
        assert!(m.swap(1, 2) == None);
        assert!(m.swap(1, 3) == Some(2));
        assert!(m.swap(1, 4) == Some(3));
    }

    #[test]
    fn test_pop() {
        let mut m = TrieMap::new();
        m.insert(1, 2);
        assert!(m.pop(&1) == Some(2));
        assert!(m.pop(&1) == None);
    }
}
