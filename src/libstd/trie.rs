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
use iterator::{FromIterator, Extendable};
use uint;
use util::{swap, replace};
use vec;

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

    /// Get an iterator over the key-value pairs in the map
    pub fn iter<'a>(&'a self) -> TrieMapIterator<'a, T> {
        TrieMapIterator {
            stack: ~[self.root.children.iter()],
            remaining_min: self.length,
            remaining_max: self.length
        }
    }

    // If `upper` is true then returns upper_bound else returns lower_bound.
    #[inline]
    fn bound_iter<'a>(&'a self, key: uint, upper: bool) -> TrieMapIterator<'a, T> {
        let mut node: &'a TrieNode<T> = &self.root;
        let mut idx = 0;
        let mut it = TrieMapIterator {
            stack: ~[],
            remaining_min: 0,
            remaining_max: self.length
        };
        loop {
            let children = &node.children;
            let child_id = chunk(key, idx);
            match children[child_id] {
                Internal(ref n) => {
                    node = &**n;
                    it.stack.push(children.slice_from(child_id + 1).iter());
                }
                External(stored, _) => {
                    if stored < key || (upper && stored == key) {
                        it.stack.push(children.slice_from(child_id + 1).iter());
                    } else {
                        it.stack.push(children.slice_from(child_id).iter());
                    }
                    return it;
                }
                Nothing => {
                    it.stack.push(children.slice_from(child_id + 1).iter());
                    return it
                }
            }
            idx += 1;
        }
    }

    /// Get an iterator pointing to the first key-value pair whose key is not less than `key`.
    /// If all keys in the map are less than `key` an empty iterator is returned.
    pub fn lower_bound_iter<'a>(&'a self, key: uint) -> TrieMapIterator<'a, T> {
        self.bound_iter(key, false)
    }

    /// Get an iterator pointing to the first key-value pair whose key is greater than `key`.
    /// If all keys in the map are not greater than `key` an empty iterator is returned.
    pub fn upper_bound_iter<'a>(&'a self, key: uint) -> TrieMapIterator<'a, T> {
        self.bound_iter(key, true)
    }
}

impl<T, Iter: Iterator<(uint, T)>> FromIterator<(uint, T), Iter> for TrieMap<T> {
    fn from_iterator(iter: &mut Iter) -> TrieMap<T> {
        let mut map = TrieMap::new();
        map.extend(iter);
        map
    }
}

impl<T, Iter: Iterator<(uint, T)>> Extendable<(uint, T), Iter> for TrieMap<T> {
    fn extend(&mut self, iter: &mut Iter) {
        for (k, v) in *iter {
            self.insert(k, v);
        }
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

    /// Get an iterator over the values in the set
    #[inline]
    pub fn iter<'a>(&'a self) -> TrieSetIterator<'a> {
        TrieSetIterator{iter: self.map.iter()}
    }

    /// Get an iterator pointing to the first value that is not less than `val`.
    /// If all values in the set are less than `val` an empty iterator is returned.
    pub fn lower_bound_iter<'a>(&'a self, val: uint) -> TrieSetIterator<'a> {
        TrieSetIterator{iter: self.map.lower_bound_iter(val)}
    }

    /// Get an iterator pointing to the first value that key is greater than `val`.
    /// If all values in the set are not greater than `val` an empty iterator is returned.
    pub fn upper_bound_iter<'a>(&'a self, val: uint) -> TrieSetIterator<'a> {
        TrieSetIterator{iter: self.map.upper_bound_iter(val)}
    }
}

impl<Iter: Iterator<uint>> FromIterator<uint, Iter> for TrieSet {
    fn from_iterator(iter: &mut Iter) -> TrieSet {
        let mut set = TrieSet::new();
        set.extend(iter);
        set
    }
}

impl<Iter: Iterator<uint>> Extendable<uint, Iter> for TrieSet {
    fn extend(&mut self, iter: &mut Iter) {
        for elem in *iter {
            self.insert(elem);
        }
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
        for elt in self.children.iter() {
            match *elt {
                Internal(ref x) => if !x.each(|i,t| f(i,t)) { return false },
                External(k, ref v) => if !f(&k, v) { return false },
                Nothing => ()
            }
        }
        true
    }

    fn each_reverse<'a>(&'a self, f: &fn(&uint, &'a T) -> bool) -> bool {
        for elt in self.children.rev_iter() {
            match *elt {
                Internal(ref x) => if !x.each_reverse(|i,t| f(i,t)) { return false },
                External(k, ref v) => if !f(&k, v) { return false },
                Nothing => ()
            }
        }
        true
    }

    fn mutate_values<'a>(&'a mut self, f: &fn(&uint, &mut T) -> bool) -> bool {
        for child in self.children.mut_iter() {
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

/// Forward iterator over a map
pub struct TrieMapIterator<'self, T> {
    priv stack: ~[vec::VecIterator<'self, Child<T>>],
    priv remaining_min: uint,
    priv remaining_max: uint
}

impl<'self, T> Iterator<(uint, &'self T)> for TrieMapIterator<'self, T> {
    fn next(&mut self) -> Option<(uint, &'self T)> {
        while !self.stack.is_empty() {
            match self.stack[self.stack.len() - 1].next() {
                None => {
                    self.stack.pop();
                }
                Some(ref child) => {
                    match **child {
                        Internal(ref node) => {
                            self.stack.push(node.children.iter());
                        }
                        External(key, ref value) => {
                            self.remaining_max -= 1;
                            if self.remaining_min > 0 {
                                self.remaining_min -= 1;
                            }
                            return Some((key, value));
                        }
                        Nothing => {}
                    }
                }
            }
        }
        return None;
    }

    #[inline]
    fn size_hint(&self) -> (uint, Option<uint>) {
        (self.remaining_min, Some(self.remaining_max))
    }
}

/// Forward iterator over a set
pub struct TrieSetIterator<'self> {
    priv iter: TrieMapIterator<'self, ()>
}

impl<'self> Iterator<uint> for TrieSetIterator<'self> {
    fn next(&mut self) -> Option<uint> {
        do self.iter.next().map |&(key, _)| { key }
    }

    fn size_hint(&self) -> (uint, Option<uint>) {
        self.iter.size_hint()
    }
}

#[cfg(test)]
pub fn check_integrity<T>(trie: &TrieNode<T>) {
    assert!(trie.count != 0);

    let mut sum = 0;

    for x in trie.children.iter() {
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
    use prelude::*;
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

        do uint::range_step(1, n, 2) |x| {
            assert!(trie.insert(x, x + 1));
            assert!(trie.contains_key(&x));
            check_integrity(&trie.root);
            true
        };

        do uint::range_step(0, n, 2) |x| {
            assert!(!trie.contains_key(&x));
            assert!(trie.insert(x, x + 1));
            check_integrity(&trie.root);
            true
        };

        for x in range(0u, n) {
            assert!(trie.contains_key(&x));
            assert!(!trie.insert(x, x + 1));
            check_integrity(&trie.root);
        }

        do uint::range_step(1, n, 2) |x| {
            assert!(trie.remove(&x));
            assert!(!trie.contains_key(&x));
            check_integrity(&trie.root);
            true
        };

        do uint::range_step(0, n, 2) |x| {
            assert!(trie.contains_key(&x));
            assert!(!trie.insert(x, x + 1));
            check_integrity(&trie.root);
            true
        };
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
        do m.each |k, v| {
            assert_eq!(*k, n);
            assert_eq!(*v, n * 2);
            n += 1;
            true
        };
    }

    #[test]
    fn test_each_break() {
        let mut m = TrieMap::new();

        for x in range(uint::max_value - 10000, uint::max_value).invert() {
            m.insert(x, x / 2);
        }

        let mut n = uint::max_value - 10000;
        do m.each |k, v| {
            if n == uint::max_value - 5000 { false } else {
                assert!(n < uint::max_value - 5000);

                assert_eq!(*k, n);
                assert_eq!(*v, n / 2);
                n += 1;
                true
            }
        };
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
        do m.each_reverse |k, v| {
            assert_eq!(*k, n);
            assert_eq!(*v, n * 2);
            n -= 1;
            true
        };
    }

    #[test]
    fn test_each_reverse_break() {
        let mut m = TrieMap::new();

        for x in range(uint::max_value - 10000, uint::max_value).invert() {
            m.insert(x, x / 2);
        }

        let mut n = uint::max_value - 1;
        do m.each_reverse |k, v| {
            if n == uint::max_value - 5000 { false } else {
                assert!(n > uint::max_value - 5000);

                assert_eq!(*k, n);
                assert_eq!(*v, n / 2);
                n -= 1;
                true
            }
        };
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

        let map: TrieMap<int> = xs.iter().map(|&x| x).collect();

        for &(k, v) in xs.iter() {
            assert_eq!(map.find(&k), Some(&v));
        }
    }

    #[test]
    fn test_iteration() {
        let empty_map : TrieMap<uint> = TrieMap::new();
        assert_eq!(empty_map.iter().next(), None);

        let first = uint::max_value - 10000;
        let last = uint::max_value;

        let mut map = TrieMap::new();
        for x in range(first, last).invert() {
            map.insert(x, x / 2);
        }

        let mut i = 0;
        for (k, &v) in map.iter() {
            assert_eq!(k, first + i);
            assert_eq!(v, k / 2);
            i += 1;
        }
        assert_eq!(i, last - first);
    }

    #[test]
    fn test_bound_iter() {
        let empty_map : TrieMap<uint> = TrieMap::new();
        assert_eq!(empty_map.lower_bound_iter(0).next(), None);
        assert_eq!(empty_map.upper_bound_iter(0).next(), None);

        let last = 999u;
        let step = 3u;
        let value = 42u;

        let mut map : TrieMap<uint> = TrieMap::new();
        do uint::range_step(0u, last, step as int) |x| {
            assert!(x % step == 0);
            map.insert(x, value);
            true
        };

        for i in range(0u, last - step) {
            let mut lb = map.lower_bound_iter(i);
            let mut ub = map.upper_bound_iter(i);
            let next_key = i - i % step + step;
            let next_pair = (next_key, &value);
            if (i % step == 0) {
                assert_eq!(lb.next(), Some((i, &value)));
            } else {
                assert_eq!(lb.next(), Some(next_pair));
            }
            assert_eq!(ub.next(), Some(next_pair));
        }

        let mut lb = map.lower_bound_iter(last - step);
        assert_eq!(lb.next(), Some((last - step, &value)));
        let mut ub = map.upper_bound_iter(last - step);
        assert_eq!(ub.next(), None);

        for i in range(last - step + 1, last) {
            let mut lb = map.lower_bound_iter(i);
            assert_eq!(lb.next(), None);
            let mut ub = map.upper_bound_iter(i);
            assert_eq!(ub.next(), None);
        }
    }
}

#[cfg(test)]
mod test_set {
    use super::*;
    use prelude::*;
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

        do trie.each |x| {
            assert_eq!(expected[i], *x);
            i += 1;
            true
        };
    }

    #[test]
    fn test_from_iter() {
        let xs = ~[9u, 8, 7, 6, 5, 4, 3, 2, 1];

        let set: TrieSet = xs.iter().map(|&x| x).collect();

        for x in xs.iter() {
            assert!(set.contains(x));
        }
    }
}
