// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! A radix trie for storing integers in sorted order

use prelude::*;

// FIXME: #3469: need to manually update TrieNode when SHIFT changes
const SHIFT: uint = 4;
const SIZE: uint = 1 << SHIFT;
const MASK: uint = SIZE - 1;

enum Child<T> {
    Internal(~TrieNode<T>),
    External(uint, T),
    Nothing
}

pub struct TrieMap<T> {
    priv root: TrieNode<T>,
    priv length: uint
}

impl<T> BaseIter<(uint, &T)> for TrieMap<T> {
    /// Visit all key-value pairs in order
    #[inline(always)]
    pure fn each(&self, f: fn(&(uint, &self/T)) -> bool) {
        self.root.each(f)
    }
    #[inline(always)]
    pure fn size_hint(&self) -> Option<uint> { Some(self.len()) }
}

impl<T> ReverseIter<(uint, &T)> for TrieMap<T> {
    /// Visit all key-value pairs in reverse order
    #[inline(always)]
    pure fn each_reverse(&self, f: fn(&(uint, &self/T)) -> bool) {
        self.root.each_reverse(f)
    }
}

impl<T> Container for TrieMap<T> {
    /// Return the number of elements in the map
    #[inline(always)]
    pure fn len(&self) -> uint { self.length }

    /// Return true if the map contains no elements
    #[inline(always)]
    pure fn is_empty(&self) -> bool { self.len() == 0 }
}

impl<T: Copy> Mutable for TrieMap<T> {
    /// Clear the map, removing all values.
    #[inline(always)]
    fn clear(&mut self) {
        self.root = TrieNode::new();
        self.length = 0;
    }
}

impl<T: Copy> Map<uint, T> for TrieMap<T> {
    /// Return true if the map contains a value for the specified key
    #[inline(always)]
    pure fn contains_key(&self, key: &uint) -> bool {
        self.find(key).is_some()
    }

    /// Visit all keys in order
    #[inline(always)]
    pure fn each_key(&self, f: fn(&uint) -> bool) {
        self.each(|&(k, _)| f(&k))
    }

    /// Visit all values in order
    #[inline(always)]
    pure fn each_value(&self, f: fn(&T) -> bool) { self.each(|&(_, v)| f(v)) }

    /// Return the value corresponding to the key in the map
    #[inline(hint)]
    pure fn find(&self, key: &uint) -> Option<&self/T> {
        let mut node: &self/TrieNode<T> = &self.root;
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

    /// Insert a key-value pair into the map. An existing value for a
    /// key is replaced by the new value. Return true if the key did
    /// not already exist in the map.
    #[inline(always)]
    fn insert(&mut self, key: uint, value: T) -> bool {
        let ret = insert(&mut self.root.count,
                         &mut self.root.children[chunk(key, 0)],
                         key, value, 1);
        if ret { self.length += 1 }
        ret
    }

    /// Remove a key-value pair from the map. Return true if the key
    /// was present in the map, otherwise false.
    #[inline(always)]
    fn remove(&mut self, key: &uint) -> bool {
        let ret = remove(&mut self.root.count,
                         &mut self.root.children[chunk(*key, 0)],
                         *key, 1);
        if ret { self.length -= 1 }
        ret
    }
}

impl<T: Copy> TrieMap<T> {
    #[inline(always)]
    static pure fn new() -> TrieMap<T> {
        TrieMap{root: TrieNode::new(), length: 0}
    }
}

impl<T> TrieMap<T> {
    /// Visit all keys in reverse order
    #[inline(always)]
    pure fn each_key_reverse(&self, f: fn(&uint) -> bool) {
        self.each_reverse(|&(k, _)| f(&k))
    }

    /// Visit all values in reverse order
    #[inline(always)]
    pure fn each_value_reverse(&self, f: fn(&T) -> bool) {
        self.each_reverse(|&(_, v)| f(v))
    }

    /// Iterate over the map and mutate the contained values
    fn mutate_values(&mut self, f: fn(uint, &mut T) -> bool) {
        self.root.mutate_values(f)
    }
}

pub struct TrieSet {
    priv map: TrieMap<()>
}

impl BaseIter<uint> for TrieSet {
    /// Visit all values in order
    pure fn each(&self, f: fn(&uint) -> bool) { self.map.each_key(f) }
    pure fn size_hint(&self) -> Option<uint> { Some(self.len()) }
}

impl ReverseIter<uint> for TrieSet {
    /// Visit all values in reverse order
    pure fn each_reverse(&self, f: fn(&uint) -> bool) {
        self.map.each_key_reverse(f)
    }
}

impl Container for TrieSet {
    /// Return the number of elements in the set
    #[inline(always)]
    pure fn len(&self) -> uint { self.map.len() }

    /// Return true if the set contains no elements
    #[inline(always)]
    pure fn is_empty(&self) -> bool { self.map.is_empty() }
}

impl Mutable for TrieSet {
    /// Clear the set, removing all values.
    #[inline(always)]
    fn clear(&mut self) { self.map.clear() }
}

impl TrieSet {
    /// Return true if the set contains a value
    #[inline(always)]
    pure fn contains(&self, value: &uint) -> bool {
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
    children: [Child<T> * 16] // FIXME: #3469: can't use the SIZE constant yet
}

impl<T: Copy> TrieNode<T> {
    #[inline(always)]
    static pure fn new() -> TrieNode<T> {
        TrieNode{count: 0, children: [Nothing, ..SIZE]}
    }
}

impl<T> TrieNode<T> {
    pure fn each(&self, f: fn(&(uint, &self/T)) -> bool) {
        for uint::range(0, self.children.len()) |idx| {
            match self.children[idx] {
                Internal(ref x) => x.each(f),
                External(k, ref v) => if !f(&(k, v)) { return },
                Nothing => ()
            }
        }
    }

    pure fn each_reverse(&self, f: fn(&(uint, &self/T)) -> bool) {
        for uint::range_rev(self.children.len(), 0) |idx| {
            match self.children[idx - 1] {
                Internal(ref x) => x.each(f),
                External(k, ref v) => if !f(&(k, v)) { return },
                Nothing => ()
            }
        }
    }

    fn mutate_values(&mut self, f: fn(uint, &mut T) -> bool) {
        for vec::each_mut(self.children) |child| {
            match *child {
                Internal(ref mut x) => x.mutate_values(f),
                External(k, ref mut v) => if !f(k, v) { return },
                Nothing => ()
            }
        }
    }
}

// if this was done via a trait, the key could be generic
#[inline(always)]
pure fn chunk(n: uint, idx: uint) -> uint {
    let real_idx = uint::bytes - 1 - idx;
    (n >> (SHIFT * real_idx)) & MASK
}

fn insert<T: Copy>(count: &mut uint, child: &mut Child<T>, key: uint,
                   value: T, idx: uint) -> bool {
    match *child {
      External(stored_key, stored_value) => {
          if stored_key == key {
              false // already in the trie
          } else {
              // conflict - split the node
              let mut new = ~TrieNode::new();
              insert(&mut new.count,
                     &mut new.children[chunk(stored_key, idx)],
                     stored_key, stored_value, idx + 1);
              insert(&mut new.count, &mut new.children[chunk(key, idx)], key,
                     value, idx + 1);
              *child = Internal(new);
              true
          }
      }
      Internal(ref mut x) => {
        insert(&mut x.count, &mut x.children[chunk(key, idx)], key, value,
               idx + 1)
      }
      Nothing => {
        *count += 1;
        *child = External(key, value);
        true
      }
    }
}

fn remove<T>(count: &mut uint, child: &mut Child<T>, key: uint,
             idx: uint) -> bool {
    let (ret, this) = match *child {
      External(stored, _) => {
          if stored == key { (true, true) } else { (false, false) }
      }
      Internal(ref mut x) => {
          let ret = remove(&mut x.count, &mut x.children[chunk(key, idx)],
                           key, idx + 1);
          (ret, x.count == 0)
      }
      Nothing => (false, false)
    };

    if this {
        *child = Nothing;
        *count -= 1;
    }
    ret
}

#[cfg(test)]
pub fn check_integrity<T>(trie: &TrieNode<T>) {
    assert trie.count != 0;

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

    assert sum == trie.count;
}

#[cfg(test)]
mod tests {
    use super::*;
    use uint;

    #[test]
    fn test_step() {
        let mut trie = TrieMap::new();
        let n = 300;

        for uint::range_step(1, n, 2) |x| {
            assert trie.insert(x, x + 1);
            assert trie.contains_key(&x);
            check_integrity(&trie.root);
        }

        for uint::range_step(0, n, 2) |x| {
            assert !trie.contains_key(&x);
            assert trie.insert(x, x + 1);
            check_integrity(&trie.root);
        }

        for uint::range(0, n) |x| {
            assert trie.contains_key(&x);
            assert !trie.insert(x, x + 1);
            check_integrity(&trie.root);
        }

        for uint::range_step(1, n, 2) |x| {
            assert trie.remove(&x);
            assert !trie.contains_key(&x);
            check_integrity(&trie.root);
        }

        for uint::range_step(0, n, 2) |x| {
            assert trie.contains_key(&x);
            assert !trie.insert(x, x + 1);
            check_integrity(&trie.root);
        }
    }
}
