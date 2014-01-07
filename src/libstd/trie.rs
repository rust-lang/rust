// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Ordered containers with integer keys, implemented as radix tries (`TrieSet` and `TrieMap` types)

use prelude::*;
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
    pub fn each_reverse<'a>(&'a self, f: |&uint, &'a T| -> bool) -> bool {
        self.root.each_reverse(f)
    }

    /// Visit all keys in reverse order
    #[inline]
    pub fn each_key_reverse(&self, f: |&uint| -> bool) -> bool {
        self.each_reverse(|k, _| f(k))
    }

    /// Visit all values in reverse order
    #[inline]
    pub fn each_value_reverse(&self, f: |&T| -> bool) -> bool {
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

    /// Get an iterator over the key-value pairs in the map, with the
    /// ability to mutate the values.
    pub fn mut_iter<'a>(&'a mut self) -> TrieMapMutIterator<'a, T> {
        TrieMapMutIterator {
            stack: ~[self.root.children.mut_iter()],
            remaining_min: self.length,
            remaining_max: self.length
        }
    }
}

// FIXME #5846 we want to be able to choose between &x and &mut x
// (with many different `x`) below, so we need to optionally pass mut
// as a tt, but the only thing we can do with a `tt` is pass them to
// other macros, so this takes the `& <mutability> <operand>` token
// sequence and forces their evalutation as an expression. (see also
// `item!` below.)
macro_rules! addr { ($e:expr) => { $e } }

macro_rules! bound {
    ($iterator_name:ident,
     // the current treemap
     self = $this:expr,
     // the key to look for
     key = $key:expr,
     // are we looking at the upper bound?
     is_upper = $upper:expr,

     // method names for slicing/iterating.
     slice_from = $slice_from:ident,
     iter = $iter:ident,

     // see the comment on `addr!`, this is just an optional mut, but
     // there's no 0-or-1 repeats yet.
     mutability = $($mut_:tt)*) => {
        {
            // # For `mut`
            // We need an unsafe pointer here because we are borrowing
            // mutable references to the internals of each of these
            // mutable nodes, while still using the outer node.
            //
            // However, we're allowed to flaunt rustc like this because we
            // never actually modify the "shape" of the nodes. The only
            // place that mutation is can actually occur is of the actual
            // values of the TrieMap (as the return value of the
            // iterator), i.e. we can never cause a deallocation of any
            // TrieNodes so the raw pointer is always valid.
            //
            // # For non-`mut`
            // We like sharing code so much that even a little unsafe won't
            // stop us.
            let this = $this;
            let mut node = addr!(& $($mut_)* this.root as * $($mut_)* TrieNode<T>);

            let key = $key;

            let mut idx = 0;
            let mut it = $iterator_name {
                stack: ~[],
                remaining_min: 0,
                remaining_max: this.length
            };
            // this addr is necessary for the `Internal` pattern.
            addr!(loop {
                    let children = unsafe {addr!(& $($mut_)* (*node).children)};
                    let child_id = chunk(key, idx);
                    let (slice_idx, ret) = match children[child_id] {
                        Internal(ref $($mut_)* n) => {
                            node = addr!(& $($mut_)* **n as * $($mut_)* TrieNode<T>);
                            (child_id + 1, false)
                        }
                        External(stored, _) => {
                            (if stored < key || ($upper && stored == key) {
                                child_id + 1
                            } else {
                                child_id
                            }, true)
                        }
                        Nothing => {
                            (child_id + 1, true)
                        }
                    };
                    it.stack.push(children.$slice_from(slice_idx).$iter());
                    if ret { return it }
                    idx += 1;
                })
        }
    }
}

impl<T> TrieMap<T> {
    // If `upper` is true then returns upper_bound else returns lower_bound.
    #[inline]
    fn bound<'a>(&'a self, key: uint, upper: bool) -> TrieMapIterator<'a, T> {
        bound!(TrieMapIterator, self = self,
               key = key, is_upper = upper,
               slice_from = slice_from, iter = iter,
               mutability = )
    }

    /// Get an iterator pointing to the first key-value pair whose key is not less than `key`.
    /// If all keys in the map are less than `key` an empty iterator is returned.
    pub fn lower_bound<'a>(&'a self, key: uint) -> TrieMapIterator<'a, T> {
        self.bound(key, false)
    }

    /// Get an iterator pointing to the first key-value pair whose key is greater than `key`.
    /// If all keys in the map are not greater than `key` an empty iterator is returned.
    pub fn upper_bound<'a>(&'a self, key: uint) -> TrieMapIterator<'a, T> {
        self.bound(key, true)
    }
    // If `upper` is true then returns upper_bound else returns lower_bound.
    #[inline]
    fn mut_bound<'a>(&'a mut self, key: uint, upper: bool) -> TrieMapMutIterator<'a, T> {
        bound!(TrieMapMutIterator, self = self,
               key = key, is_upper = upper,
               slice_from = mut_slice_from, iter = mut_iter,
               mutability = mut)
    }

    /// Get an iterator pointing to the first key-value pair whose key is not less than `key`.
    /// If all keys in the map are less than `key` an empty iterator is returned.
    pub fn mut_lower_bound<'a>(&'a mut self, key: uint) -> TrieMapMutIterator<'a, T> {
        self.mut_bound(key, false)
    }

    /// Get an iterator pointing to the first key-value pair whose key is greater than `key`.
    /// If all keys in the map are not greater than `key` an empty iterator is returned.
    pub fn mut_upper_bound<'a>(&'a mut self, key: uint) -> TrieMapMutIterator<'a, T> {
        self.mut_bound(key, true)
    }
}

impl<T> FromIterator<(uint, T)> for TrieMap<T> {
    fn from_iterator<Iter: Iterator<(uint, T)>>(iter: &mut Iter) -> TrieMap<T> {
        let mut map = TrieMap::new();
        map.extend(iter);
        map
    }
}

impl<T> Extendable<(uint, T)> for TrieMap<T> {
    fn extend<Iter: Iterator<(uint, T)>>(&mut self, iter: &mut Iter) {
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

    /// Visit all values in reverse order
    #[inline]
    pub fn each_reverse(&self, f: |&uint| -> bool) -> bool {
        self.map.each_key_reverse(f)
    }

    /// Get an iterator over the values in the set
    #[inline]
    pub fn iter<'a>(&'a self) -> TrieSetIterator<'a> {
        TrieSetIterator{iter: self.map.iter()}
    }

    /// Get an iterator pointing to the first value that is not less than `val`.
    /// If all values in the set are less than `val` an empty iterator is returned.
    pub fn lower_bound<'a>(&'a self, val: uint) -> TrieSetIterator<'a> {
        TrieSetIterator{iter: self.map.lower_bound(val)}
    }

    /// Get an iterator pointing to the first value that key is greater than `val`.
    /// If all values in the set are not greater than `val` an empty iterator is returned.
    pub fn upper_bound<'a>(&'a self, val: uint) -> TrieSetIterator<'a> {
        TrieSetIterator{iter: self.map.upper_bound(val)}
    }
}

impl FromIterator<uint> for TrieSet {
    fn from_iterator<Iter: Iterator<uint>>(iter: &mut Iter) -> TrieSet {
        let mut set = TrieSet::new();
        set.extend(iter);
        set
    }
}

impl Extendable<uint> for TrieSet {
    fn extend<Iter: Iterator<uint>>(&mut self, iter: &mut Iter) {
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
    fn each_reverse<'a>(&'a self, f: |&uint, &'a T| -> bool) -> bool {
        for elt in self.children.rev_iter() {
            match *elt {
                Internal(ref x) => if !x.each_reverse(|i,t| f(i,t)) { return false },
                External(k, ref v) => if !f(&k, v) { return false },
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
        External(stored, ref mut value) if stored == key => Some(value),
        External(..) => None,
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
      External(..) => (None, false),
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
pub struct TrieMapIterator<'a, T> {
    priv stack: ~[vec::VecIterator<'a, Child<T>>],
    priv remaining_min: uint,
    priv remaining_max: uint
}

/// Forward iterator over the key-value pairs of a map, with the
/// values being mutable.
pub struct TrieMapMutIterator<'a, T> {
    priv stack: ~[vec::VecMutIterator<'a, Child<T>>],
    priv remaining_min: uint,
    priv remaining_max: uint
}

// FIXME #5846: see `addr!` above.
macro_rules! item { ($i:item) => {$i}}

macro_rules! iterator_impl {
    ($name:ident,
     iter = $iter:ident,
     mutability = $($mut_:tt)*) => {
        item!(impl<'a, T> Iterator<(uint, &'a $($mut_)* T)> for $name<'a, T> {
                fn next(&mut self) -> Option<(uint, &'a $($mut_)* T)> {
                    while !self.stack.is_empty() {
                        match self.stack[self.stack.len() - 1].next() {
                            None => {
                                self.stack.pop();
                            }
                            Some(child) => {
                                addr!(match *child {
                                        Internal(ref $($mut_)* node) => {
                                            self.stack.push(node.children.$iter());
                                        }
                                        External(key, ref $($mut_)* value) => {
                                            self.remaining_max -= 1;
                                            if self.remaining_min > 0 {
                                                self.remaining_min -= 1;
                                            }
                                            return Some((key, value));
                                        }
                                        Nothing => {}
                                    })
                            }
                        }
                    }
                    return None;
                }

                #[inline]
                fn size_hint(&self) -> (uint, Option<uint>) {
                    (self.remaining_min, Some(self.remaining_max))
                }
            })
    }
}

iterator_impl! { TrieMapIterator, iter = iter, mutability = }
iterator_impl! { TrieMapMutIterator, iter = mut_iter, mutability = mut }

/// Forward iterator over a set
pub struct TrieSetIterator<'a> {
    priv iter: TrieMapIterator<'a, ()>
}

impl<'a> Iterator<uint> for TrieSetIterator<'a> {
    fn next(&mut self) -> Option<uint> {
        self.iter.next().map(|(key, _)| key)
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
    use iter::range_step;
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
    fn test_find_mut_missing() {
        let mut m = TrieMap::new();
        assert!(m.find_mut(&0).is_none());
        assert!(m.insert(1, 12));
        assert!(m.find_mut(&0).is_none());
        assert!(m.insert(2, 8));
        assert!(m.find_mut(&0).is_none());
    }

    #[test]
    fn test_step() {
        let mut trie = TrieMap::new();
        let n = 300u;

        for x in range_step(1u, n, 2) {
            assert!(trie.insert(x, x + 1));
            assert!(trie.contains_key(&x));
            check_integrity(&trie.root);
        }

        for x in range_step(0u, n, 2) {
            assert!(!trie.contains_key(&x));
            assert!(trie.insert(x, x + 1));
            check_integrity(&trie.root);
        }

        for x in range(0u, n) {
            assert!(trie.contains_key(&x));
            assert!(!trie.insert(x, x + 1));
            check_integrity(&trie.root);
        }

        for x in range_step(1u, n, 2) {
            assert!(trie.remove(&x));
            assert!(!trie.contains_key(&x));
            check_integrity(&trie.root);
        }

        for x in range_step(0u, n, 2) {
            assert!(trie.contains_key(&x));
            assert!(!trie.insert(x, x + 1));
            check_integrity(&trie.root);
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
        m.each_reverse(|k, v| {
            assert_eq!(*k, n);
            assert_eq!(*v, n * 2);
            n -= 1;
            true
        });
    }

    #[test]
    fn test_each_reverse_break() {
        let mut m = TrieMap::new();

        for x in range(uint::max_value - 10000, uint::max_value).invert() {
            m.insert(x, x / 2);
        }

        let mut n = uint::max_value - 1;
        m.each_reverse(|k, v| {
            if n == uint::max_value - 5000 { false } else {
                assert!(n > uint::max_value - 5000);

                assert_eq!(*k, n);
                assert_eq!(*v, n / 2);
                n -= 1;
                true
            }
        });
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
    fn test_mut_iter() {
        let mut empty_map : TrieMap<uint> = TrieMap::new();
        assert!(empty_map.mut_iter().next().is_none());

        let first = uint::max_value - 10000;
        let last = uint::max_value;

        let mut map = TrieMap::new();
        for x in range(first, last).invert() {
            map.insert(x, x / 2);
        }

        let mut i = 0;
        for (k, v) in map.mut_iter() {
            assert_eq!(k, first + i);
            *v -= k / 2;
            i += 1;
        }
        assert_eq!(i, last - first);

        assert!(map.iter().all(|(_, &v)| v == 0));
    }

    #[test]
    fn test_bound() {
        let empty_map : TrieMap<uint> = TrieMap::new();
        assert_eq!(empty_map.lower_bound(0).next(), None);
        assert_eq!(empty_map.upper_bound(0).next(), None);

        let last = 999u;
        let step = 3u;
        let value = 42u;

        let mut map : TrieMap<uint> = TrieMap::new();
        for x in range_step(0u, last, step) {
            assert!(x % step == 0);
            map.insert(x, value);
        }

        for i in range(0u, last - step) {
            let mut lb = map.lower_bound(i);
            let mut ub = map.upper_bound(i);
            let next_key = i - i % step + step;
            let next_pair = (next_key, &value);
            if (i % step == 0) {
                assert_eq!(lb.next(), Some((i, &value)));
            } else {
                assert_eq!(lb.next(), Some(next_pair));
            }
            assert_eq!(ub.next(), Some(next_pair));
        }

        let mut lb = map.lower_bound(last - step);
        assert_eq!(lb.next(), Some((last - step, &value)));
        let mut ub = map.upper_bound(last - step);
        assert_eq!(ub.next(), None);

        for i in range(last - step + 1, last) {
            let mut lb = map.lower_bound(i);
            assert_eq!(lb.next(), None);
            let mut ub = map.upper_bound(i);
            assert_eq!(ub.next(), None);
        }
    }

    #[test]
    fn test_mut_bound() {
        let empty_map : TrieMap<uint> = TrieMap::new();
        assert_eq!(empty_map.lower_bound(0).next(), None);
        assert_eq!(empty_map.upper_bound(0).next(), None);

        let mut m_lower = TrieMap::new();
        let mut m_upper = TrieMap::new();
        for i in range(0u, 100) {
            m_lower.insert(2 * i, 4 * i);
            m_upper.insert(2 * i, 4 * i);
        }

        for i in range(0u, 199) {
            let mut lb_it = m_lower.mut_lower_bound(i);
            let (k, v) = lb_it.next().unwrap();
            let lb = i + i % 2;
            assert_eq!(lb, k);
            *v -= k;
        }

        for i in range(0u, 198) {
            let mut ub_it = m_upper.mut_upper_bound(i);
            let (k, v) = ub_it.next().unwrap();
            let ub = i + 2 - i % 2;
            assert_eq!(ub, k);
            *v -= k;
        }

        assert!(m_lower.mut_lower_bound(199).next().is_none());
        assert!(m_upper.mut_upper_bound(198).next().is_none());

        assert!(m_lower.iter().all(|(_, &x)| x == 0));
        assert!(m_upper.iter().all(|(_, &x)| x == 0));
    }
}

#[cfg(test)]
mod bench_map {
    use super::*;
    use prelude::*;
    use rand::{weak_rng, Rng};
    use extra::test::BenchHarness;

    #[bench]
    fn bench_iter_small(bh: &mut BenchHarness) {
        let mut m = TrieMap::<uint>::new();
        let mut rng = weak_rng();
        for _ in range(0, 20) {
            m.insert(rng.gen(), rng.gen());
        }

        bh.iter(|| for _ in m.iter() {})
    }

    #[bench]
    fn bench_iter_large(bh: &mut BenchHarness) {
        let mut m = TrieMap::<uint>::new();
        let mut rng = weak_rng();
        for _ in range(0, 1000) {
            m.insert(rng.gen(), rng.gen());
        }

        bh.iter(|| for _ in m.iter() {})
    }

    #[bench]
    fn bench_lower_bound(bh: &mut BenchHarness) {
        let mut m = TrieMap::<uint>::new();
        let mut rng = weak_rng();
        for _ in range(0, 1000) {
            m.insert(rng.gen(), rng.gen());
        }

        bh.iter(|| {
                for _ in range(0, 10) {
                    m.lower_bound(rng.gen());
                }
            });
    }

    #[bench]
    fn bench_upper_bound(bh: &mut BenchHarness) {
        let mut m = TrieMap::<uint>::new();
        let mut rng = weak_rng();
        for _ in range(0, 1000) {
            m.insert(rng.gen(), rng.gen());
        }

        bh.iter(|| {
                for _ in range(0, 10) {
                    m.upper_bound(rng.gen());
                }
            });
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

        for (i, x) in trie.iter().enumerate() {
            assert_eq!(expected[i], x);
        }
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
