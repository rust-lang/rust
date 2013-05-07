// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! An ordered map and set implemented as self-balancing binary search
//! trees. The only requirement for the types is that the key implements
//! `TotalOrd`.

use core::iterator::*;
use core::util::replace;

// This is implemented as an AA tree, which is a simplified variation of
// a red-black tree where where red (horizontal) nodes can only be added
// as a right child. The time complexity is the same, and re-balancing
// operations are more frequent but also cheaper.

// Future improvements:

// range search - O(log n) retrieval of an iterator from some key

// (possibly) implement the overloads Python does for sets:
//   * intersection: &
//   * difference: -
//   * symmetric difference: ^
//   * union: |
// These would be convenient since the methods work like `each`

pub struct TreeMap<K, V> {
    priv root: Option<~TreeNode<K, V>>,
    priv length: uint
}

impl<K: Eq + TotalOrd, V: Eq> Eq for TreeMap<K, V> {
    fn eq(&self, other: &TreeMap<K, V>) -> bool {
        if self.len() != other.len() {
            false
        } else {
            let mut x = self.iter();
            let mut y = other.iter();
            for self.len().times {
                if x.next().unwrap() != y.next().unwrap() {
                    return false
                }
            }
            true
        }
    }
    fn ne(&self, other: &TreeMap<K, V>) -> bool { !self.eq(other) }
}

// Lexicographical comparison
fn lt<K: Ord + TotalOrd, V>(a: &TreeMap<K, V>,
                                 b: &TreeMap<K, V>) -> bool {
    let mut x = a.iter();
    let mut y = b.iter();

    let (a_len, b_len) = (a.len(), b.len());
    for uint::min(a_len, b_len).times {
        let (key_a,_) = x.next().unwrap();
        let (key_b,_) = y.next().unwrap();
        if *key_a < *key_b { return true; }
        if *key_a > *key_b { return false; }
    };

    a_len < b_len
}

impl<K: Ord + TotalOrd, V> Ord for TreeMap<K, V> {
    #[inline(always)]
    fn lt(&self, other: &TreeMap<K, V>) -> bool { lt(self, other) }
    #[inline(always)]
    fn le(&self, other: &TreeMap<K, V>) -> bool { !lt(other, self) }
    #[inline(always)]
    fn ge(&self, other: &TreeMap<K, V>) -> bool { !lt(self, other) }
    #[inline(always)]
    fn gt(&self, other: &TreeMap<K, V>) -> bool { lt(other, self) }
}

impl<K: TotalOrd, V> Container for TreeMap<K, V> {
    /// Return the number of elements in the map
    fn len(&const self) -> uint { self.length }

    /// Return true if the map contains no elements
    fn is_empty(&const self) -> bool { self.root.is_none() }
}

impl<K: TotalOrd, V> Mutable for TreeMap<K, V> {
    /// Clear the map, removing all key-value pairs.
    fn clear(&mut self) {
        self.root = None;
        self.length = 0
    }
}

impl<K: TotalOrd, V> Map<K, V> for TreeMap<K, V> {
    /// Return true if the map contains a value for the specified key
    fn contains_key(&self, key: &K) -> bool {
        self.find(key).is_some()
    }

    /// Visit all key-value pairs in order
    fn each<'a>(&'a self, f: &fn(&'a K, &'a V) -> bool) {
        each(&self.root, f)
    }

    /// Visit all keys in order
    fn each_key(&self, f: &fn(&K) -> bool) {
        self.each(|k, _| f(k))
    }

    /// Visit all values in order
    fn each_value<'a>(&'a self, f: &fn(&'a V) -> bool) {
        self.each(|_, v| f(v))
    }

    /// Iterate over the map and mutate the contained values
    fn mutate_values(&mut self, f: &fn(&K, &mut V) -> bool) {
        mutate_values(&mut self.root, f);
    }

    /// Return a reference to the value corresponding to the key
    fn find<'a>(&'a self, key: &K) -> Option<&'a V> {
        let mut current: &'a Option<~TreeNode<K, V>> = &self.root;
        loop {
            match *current {
              Some(ref r) => {
                match key.cmp(&r.key) {
                  Less => current = &r.left,
                  Greater => current = &r.right,
                  Equal => return Some(&r.value)
                }
              }
              None => return None
            }
        }
    }

    /// Return a mutable reference to the value corresponding to the key
    #[inline(always)]
    fn find_mut<'a>(&'a mut self, key: &K) -> Option<&'a mut V> {
        find_mut(&mut self.root, key)
    }

    /// Insert a key-value pair into the map. An existing value for a
    /// key is replaced by the new value. Return true if the key did
    /// not already exist in the map.
    fn insert(&mut self, key: K, value: V) -> bool {
        self.swap(key, value).is_none()
    }

    /// Remove a key-value pair from the map. Return true if the key
    /// was present in the map, otherwise false.
    fn remove(&mut self, key: &K) -> bool {
        self.pop(key).is_some()
    }

    /// Insert a key-value pair from the map. If the key already had a value
    /// present in the map, that value is returned. Otherwise None is returned.
    fn swap(&mut self, key: K, value: V) -> Option<V> {
        let ret = insert(&mut self.root, key, value);
        if ret.is_none() { self.length += 1 }
        ret
    }

    /// Removes a key from the map, returning the value at the key if the key
    /// was previously in the map.
    fn pop(&mut self, key: &K) -> Option<V> {
        let ret = remove(&mut self.root, key);
        if ret.is_some() { self.length -= 1 }
        ret
    }
}

pub impl<K: TotalOrd, V> TreeMap<K, V> {
    /// Create an empty TreeMap
    fn new() -> TreeMap<K, V> { TreeMap{root: None, length: 0} }

    /// Visit all key-value pairs in reverse order
    fn each_reverse<'a>(&'a self, f: &fn(&'a K, &'a V) -> bool) {
        each_reverse(&self.root, f);
    }

    /// Visit all keys in reverse order
    fn each_key_reverse(&self, f: &fn(&K) -> bool) {
        self.each_reverse(|k, _| f(k))
    }

    /// Visit all values in reverse order
    fn each_value_reverse(&self, f: &fn(&V) -> bool) {
        self.each_reverse(|_, v| f(v))
    }

    /// Get a lazy iterator over the key-value pairs in the map.
    /// Requires that it be frozen (immutable).
    fn iter<'a>(&'a self) -> TreeMapIterator<'a, K, V> {
        TreeMapIterator{stack: ~[], node: &self.root}
    }
}

/// Lazy forward iterator over a map
pub struct TreeMapIterator<'self, K, V> {
    priv stack: ~[&'self ~TreeNode<K, V>],
    priv node: &'self Option<~TreeNode<K, V>>
}

impl<'self, K, V> Iterator<(&'self K, &'self V)> for TreeMapIterator<'self, K, V> {
    /// Advance the iterator to the next node (in order) and return a
    /// tuple with a reference to the key and value. If there are no
    /// more nodes, return `None`.
    fn next(&mut self) -> Option<(&'self K, &'self V)> {
        while !self.stack.is_empty() || self.node.is_some() {
            match *self.node {
              Some(ref x) => {
                self.stack.push(x);
                self.node = &x.left;
              }
              None => {
                let res = self.stack.pop();
                self.node = &res.right;
                return Some((&res.key, &res.value));
              }
            }
        }
        None
    }
}

impl<'self, T> Iterator<&'self T> for TreeSetIterator<'self, T> {
    /// Advance the iterator to the next node (in order). If there are no more nodes, return `None`.
    #[inline(always)]
    fn next(&mut self) -> Option<&'self T> {
        do self.iter.next().map |&(value, _)| { value }
    }
}

pub struct TreeSet<T> {
    priv map: TreeMap<T, ()>
}

impl<T: TotalOrd> BaseIter<T> for TreeSet<T> {
    /// Visit all values in order
    #[inline(always)]
    fn each(&self, f: &fn(&T) -> bool) { self.map.each_key(f) }
    #[inline(always)]
    fn size_hint(&self) -> Option<uint> { Some(self.len()) }
}

impl<T: TotalOrd> ReverseIter<T> for TreeSet<T> {
    /// Visit all values in reverse order
    #[inline(always)]
    fn each_reverse(&self, f: &fn(&T) -> bool) {
        self.map.each_key_reverse(f)
    }
}

impl<T: Eq + TotalOrd> Eq for TreeSet<T> {
    #[inline(always)]
    fn eq(&self, other: &TreeSet<T>) -> bool { self.map == other.map }
    #[inline(always)]
    fn ne(&self, other: &TreeSet<T>) -> bool { self.map != other.map }
}

impl<T: Ord + TotalOrd> Ord for TreeSet<T> {
    #[inline(always)]
    fn lt(&self, other: &TreeSet<T>) -> bool { self.map < other.map }
    #[inline(always)]
    fn le(&self, other: &TreeSet<T>) -> bool { self.map <= other.map }
    #[inline(always)]
    fn ge(&self, other: &TreeSet<T>) -> bool { self.map >= other.map }
    #[inline(always)]
    fn gt(&self, other: &TreeSet<T>) -> bool { self.map > other.map }
}

impl<T: TotalOrd> Container for TreeSet<T> {
    /// Return the number of elements in the set
    #[inline(always)]
    fn len(&const self) -> uint { self.map.len() }

    /// Return true if the set contains no elements
    #[inline(always)]
    fn is_empty(&const self) -> bool { self.map.is_empty() }
}

impl<T: TotalOrd> Mutable for TreeSet<T> {
    /// Clear the set, removing all values.
    #[inline(always)]
    fn clear(&mut self) { self.map.clear() }
}

impl<T: TotalOrd> Set<T> for TreeSet<T> {
    /// Return true if the set contains a value
    #[inline(always)]
    fn contains(&self, value: &T) -> bool {
        self.map.contains_key(value)
    }

    /// Add a value to the set. Return true if the value was not already
    /// present in the set.
    #[inline(always)]
    fn insert(&mut self, value: T) -> bool { self.map.insert(value, ()) }

    /// Remove a value from the set. Return true if the value was
    /// present in the set.
    #[inline(always)]
    fn remove(&mut self, value: &T) -> bool { self.map.remove(value) }

    /// Return true if the set has no elements in common with `other`.
    /// This is equivalent to checking for an empty intersection.
    fn is_disjoint(&self, other: &TreeSet<T>) -> bool {
        let mut x = self.iter();
        let mut y = other.iter();
        let mut a = x.next();
        let mut b = y.next();
        while a.is_some() && b.is_some() {
            let a1 = a.unwrap();
            let b1 = b.unwrap();
            match a1.cmp(b1) {
              Less => a = x.next(),
              Greater => b = y.next(),
              Equal => return false
            }
        }
        true
    }

    /// Return true if the set is a subset of another
    #[inline(always)]
    fn is_subset(&self, other: &TreeSet<T>) -> bool {
        other.is_superset(self)
    }

    /// Return true if the set is a superset of another
    fn is_superset(&self, other: &TreeSet<T>) -> bool {
        let mut x = self.iter();
        let mut y = other.iter();
        let mut a = x.next();
        let mut b = y.next();
        while b.is_some() {
            if a.is_none() {
                return false
            }

            let a1 = a.unwrap();
            let b1 = b.unwrap();

            match a1.cmp(b1) {
              Less => (),
              Greater => return false,
              Equal => b = y.next(),
            }

            a = x.next();
        }
        true
    }

    /// Visit the values (in-order) representing the difference
    fn difference(&self, other: &TreeSet<T>, f: &fn(&T) -> bool) {
        let mut x = self.iter();
        let mut y = other.iter();

        let mut a = x.next();
        let mut b = y.next();

        while a.is_some() {
            if b.is_none() {
                return do a.while_some() |a1| {
                    if f(a1) { x.next() } else { None }
                }
            }

            let a1 = a.unwrap();
            let b1 = b.unwrap();

            let cmp = a1.cmp(b1);

            if cmp == Less {
                if !f(a1) { return }
                a = x.next();
            } else {
                if cmp == Equal { a = x.next() }
                b = y.next();
            }
        }
    }

    /// Visit the values (in-order) representing the symmetric difference
    fn symmetric_difference(&self, other: &TreeSet<T>,
                                 f: &fn(&T) -> bool) {
        let mut x = self.iter();
        let mut y = other.iter();

        let mut a = x.next();
        let mut b = y.next();

        while a.is_some() {
            if b.is_none() {
                return do a.while_some() |a1| {
                    if f(a1) { x.next() } else { None }
                }
            }

            let a1 = a.unwrap();
            let b1 = b.unwrap();

            let cmp = a1.cmp(b1);

            if cmp == Less {
                if !f(a1) { return }
                a = x.next();
            } else {
                if cmp == Greater {
                    if !f(b1) { return }
                } else {
                    a = x.next();
                }
                b = y.next();
            }
        }
        do b.while_some |b1| {
            if f(b1) { y.next() } else { None }
        }
    }

    /// Visit the values (in-order) representing the intersection
    fn intersection(&self, other: &TreeSet<T>, f: &fn(&T) -> bool) {
        let mut x = self.iter();
        let mut y = other.iter();

        let mut a = x.next();
        let mut b = y.next();

        while a.is_some() && b.is_some() {
            let a1 = a.unwrap();
            let b1 = b.unwrap();

            let cmp = a1.cmp(b1);

            if cmp == Less {
                a = x.next();
            } else {
                if cmp == Equal {
                    if !f(a1) { return }
                }
                b = y.next();
            }
        }
    }

    /// Visit the values (in-order) representing the union
    fn union(&self, other: &TreeSet<T>, f: &fn(&T) -> bool) {
        let mut x = self.iter();
        let mut y = other.iter();

        let mut a = x.next();
        let mut b = y.next();

        while a.is_some() {
            if b.is_none() {
                return do a.while_some() |a1| {
                    if f(a1) { x.next() } else { None }
                }
            }

            let a1 = a.unwrap();
            let b1 = b.unwrap();

            let cmp = a1.cmp(b1);

            if cmp == Greater {
                if !f(b1) { return }
                b = y.next();
            } else {
                if !f(a1) { return }
                if cmp == Equal {
                    b = y.next();
                }
                a = x.next();
            }
        }
        do b.while_some |b1| {
            if f(b1) { y.next() } else { None }
        }
    }
}

pub impl <T: TotalOrd> TreeSet<T> {
    /// Create an empty TreeSet
    #[inline(always)]
    fn new() -> TreeSet<T> { TreeSet{map: TreeMap::new()} }

    /// Get a lazy iterator over the values in the set.
    /// Requires that it be frozen (immutable).
    #[inline(always)]
    fn iter<'a>(&'a self) -> TreeSetIterator<'a, T> {
        TreeSetIterator{iter: self.map.iter()}
    }
}

/// Lazy forward iterator over a set
pub struct TreeSetIterator<'self, T> {
    priv iter: TreeMapIterator<'self, T, ()>
}

// Nodes keep track of their level in the tree, starting at 1 in the
// leaves and with a red child sharing the level of the parent.
struct TreeNode<K, V> {
    key: K,
    value: V,
    left: Option<~TreeNode<K, V>>,
    right: Option<~TreeNode<K, V>>,
    level: uint
}

pub impl<K: TotalOrd, V> TreeNode<K, V> {
    #[inline(always)]
    fn new(key: K, value: V) -> TreeNode<K, V> {
        TreeNode{key: key, value: value, left: None, right: None, level: 1}
    }
}

fn each<'r, K: TotalOrd, V>(node: &'r Option<~TreeNode<K, V>>,
                            f: &fn(&'r K, &'r V) -> bool) {
    for node.each |x| {
        each(&x.left, f);
        if f(&x.key, &x.value) { each(&x.right, f) }
    }
}

fn each_reverse<'r, K: TotalOrd, V>(node: &'r Option<~TreeNode<K, V>>,
                                    f: &fn(&'r K, &'r V) -> bool) {
    for node.each |x| {
        each_reverse(&x.right, f);
        if f(&x.key, &x.value) { each_reverse(&x.left, f) }
    }
}

fn mutate_values<'r, K: TotalOrd, V>(node: &'r mut Option<~TreeNode<K, V>>,
                                     f: &fn(&'r K, &'r mut V) -> bool)
                                  -> bool {
    match *node {
      Some(~TreeNode{key: ref key, value: ref mut value, left: ref mut left,
                     right: ref mut right, _}) => {
        if !mutate_values(left, f) { return false }
        if !f(key, value) { return false }
        if !mutate_values(right, f) { return false }
      }
      None => return false
    }
    true
}

// Remove left horizontal link by rotating right
fn skew<K: TotalOrd, V>(node: &mut ~TreeNode<K, V>) {
    if node.left.map_default(false, |x| x.level == node.level) {
        let mut save = node.left.swap_unwrap();
        node.left <-> save.right; // save.right now None
        *node <-> save;
        node.right = Some(save);
    }
}

// Remove dual horizontal link by rotating left and increasing level of
// the parent
fn split<K: TotalOrd, V>(node: &mut ~TreeNode<K, V>) {
    if node.right.map_default(false,
      |x| x.right.map_default(false, |y| y.level == node.level)) {
        let mut save = node.right.swap_unwrap();
        node.right <-> save.left; // save.left now None
        save.level += 1;
        *node <-> save;
        node.left = Some(save);
    }
}

fn find_mut<'r, K: TotalOrd, V>(node: &'r mut Option<~TreeNode<K, V>>,
                                key: &K)
                             -> Option<&'r mut V> {
    match *node {
      Some(ref mut x) => {
        match key.cmp(&x.key) {
          Less => find_mut(&mut x.left, key),
          Greater => find_mut(&mut x.right, key),
          Equal => Some(&mut x.value),
        }
      }
      None => None
    }
}

fn insert<K: TotalOrd, V>(node: &mut Option<~TreeNode<K, V>>,
                          key: K, value: V) -> Option<V> {
    match *node {
      Some(ref mut save) => {
        match key.cmp(&save.key) {
          Less => {
            let inserted = insert(&mut save.left, key, value);
            skew(save);
            split(save);
            inserted
          }
          Greater => {
            let inserted = insert(&mut save.right, key, value);
            skew(save);
            split(save);
            inserted
          }
          Equal => {
            save.key = key;
            Some(replace(&mut save.value, value))
          }
        }
      }
      None => {
       *node = Some(~TreeNode::new(key, value));
        None
      }
    }
}

fn remove<K: TotalOrd, V>(node: &mut Option<~TreeNode<K, V>>,
                          key: &K) -> Option<V> {
    fn heir_swap<K: TotalOrd, V>(node: &mut ~TreeNode<K, V>,
                            child: &mut Option<~TreeNode<K, V>>) {
        // *could* be done without recursion, but it won't borrow check
        for child.each_mut |x| {
            if x.right.is_some() {
                heir_swap(node, &mut x.right);
            } else {
                node.key <-> x.key;
                node.value <-> x.value;
            }
        }
    }

    match *node {
      None => {
        return None; // bottom of tree
      }
      Some(ref mut save) => {
        let (ret, rebalance) = match key.cmp(&save.key) {
          Less => (remove(&mut save.left, key), true),
          Greater => (remove(&mut save.right, key), true),
          Equal => {
            if save.left.is_some() {
                if save.right.is_some() {
                    let mut left = save.left.swap_unwrap();
                    if left.right.is_some() {
                        heir_swap(save, &mut left.right);
                    } else {
                        save.key <-> left.key;
                        save.value <-> left.value;
                    }
                    save.left = Some(left);
                    (remove(&mut save.left, key), true)
                } else {
                    let new = save.left.swap_unwrap();
                    let ~TreeNode{value, _} = replace(save, new);
                    *save = save.left.swap_unwrap();
                    (Some(value), true)
                }
            } else if save.right.is_some() {
                let new = save.right.swap_unwrap();
                let ~TreeNode{value, _} = replace(save, new);
                (Some(value), true)
            } else {
                (None, false)
            }
          }
        };

        if rebalance {
            let left_level = save.left.map_default(0, |x| x.level);
            let right_level = save.right.map_default(0, |x| x.level);

            // re-balance, if necessary
            if left_level < save.level - 1 || right_level < save.level - 1 {
                save.level -= 1;

                if right_level > save.level {
                    for save.right.each_mut |x| { x.level = save.level }
                }

                skew(save);

                for save.right.each_mut |right| {
                    skew(right);
                    for right.right.each_mut |x| { skew(x) }
                }

                split(save);
                for save.right.each_mut |x| { split(x) }
            }

            return ret;
        }
      }
    }
    return match replace(node, None) {
        Some(~TreeNode{value, _}) => Some(value), None => fail!()
    };
}

#[cfg(test)]
mod test_treemap {
    use core::iterator::*;
    use super::*;
    use core::rand::RngUtil;

    #[test]
    fn find_empty() {
        let m = TreeMap::new::<int, int>(); assert!(m.find(&5) == None);
    }

    #[test]
    fn find_not_found() {
        let mut m = TreeMap::new();
        assert!(m.insert(1, 2));
        assert!(m.insert(5, 3));
        assert!(m.insert(9, 3));
        assert!(m.find(&2) == None);
    }

    #[test]
    fn test_find_mut() {
        let mut m = TreeMap::new();
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
    fn insert_replace() {
        let mut m = TreeMap::new();
        assert!(m.insert(5, 2));
        assert!(m.insert(2, 9));
        assert!(!m.insert(2, 11));
        assert!(m.find(&2).unwrap() == &11);
    }

    #[test]
    fn test_clear() {
        let mut m = TreeMap::new();
        m.clear();
        assert!(m.insert(5, 11));
        assert!(m.insert(12, -3));
        assert!(m.insert(19, 2));
        m.clear();
        assert!(m.find(&5).is_none());
        assert!(m.find(&12).is_none());
        assert!(m.find(&19).is_none());
        assert!(m.is_empty());
    }

    #[test]
    fn u8_map() {
        let mut m = TreeMap::new();

        let k1 = str::to_bytes(~"foo");
        let k2 = str::to_bytes(~"bar");
        let v1 = str::to_bytes(~"baz");
        let v2 = str::to_bytes(~"foobar");

        m.insert(copy k1, copy v1);
        m.insert(copy k2, copy v2);

        assert!(m.find(&k2) == Some(&v2));
        assert!(m.find(&k1) == Some(&v1));
    }

    fn check_equal<K: Eq + TotalOrd, V: Eq>(ctrl: &[(K, V)],
                                            map: &TreeMap<K, V>) {
        assert!(ctrl.is_empty() == map.is_empty());
        for ctrl.each |x| {
            let &(k, v) = x;
            assert!(map.find(&k).unwrap() == &v)
        }
        for map.each |map_k, map_v| {
            let mut found = false;
            for ctrl.each |x| {
                let &(ctrl_k, ctrl_v) = x;
                if *map_k == ctrl_k {
                    assert!(*map_v == ctrl_v);
                    found = true;
                    break;
                }
            }
            assert!(found);
        }
    }

    fn check_left<K: TotalOrd, V>(node: &Option<~TreeNode<K, V>>,
                                  parent: &~TreeNode<K, V>) {
        match *node {
          Some(ref r) => {
            assert!(r.key.cmp(&parent.key) == Less);
            assert!(r.level == parent.level - 1); // left is black
            check_left(&r.left, r);
            check_right(&r.right, r, false);
          }
          None => assert!(parent.level == 1) // parent is leaf
        }
    }

    fn check_right<K: TotalOrd, V>(node: &Option<~TreeNode<K, V>>,
                                   parent: &~TreeNode<K, V>,
                                   parent_red: bool) {
        match *node {
          Some(ref r) => {
            assert!(r.key.cmp(&parent.key) == Greater);
            let red = r.level == parent.level;
            if parent_red { assert!(!red) } // no dual horizontal links
            // Right red or black
            assert!(red || r.level == parent.level - 1);
            check_left(&r.left, r);
            check_right(&r.right, r, red);
          }
          None => assert!(parent.level == 1) // parent is leaf
        }
    }

    fn check_structure<K: TotalOrd, V>(map: &TreeMap<K, V>) {
        match map.root {
          Some(ref r) => {
            check_left(&r.left, r);
            check_right(&r.right, r, false);
          }
          None => ()
        }
    }

    #[test]
    fn test_rand_int() {
        let mut map = TreeMap::new::<int, int>();
        let mut ctrl = ~[];

        check_equal(ctrl, &map);
        assert!(map.find(&5).is_none());

        let rng = rand::IsaacRng::new_seeded(&[42]);

        for 3.times {
            for 90.times {
                let k = rng.gen();
                let v = rng.gen();
                if !ctrl.contains(&(k, v)) {
                    assert!(map.insert(k, v));
                    ctrl.push((k, v));
                    check_structure(&map);
                    check_equal(ctrl, &map);
                }
            }

            for 30.times {
                let r = rng.gen_uint_range(0, ctrl.len());
                let (key, _) = vec::remove(&mut ctrl, r);
                assert!(map.remove(&key));
                check_structure(&map);
                check_equal(ctrl, &map);
            }
        }
    }

    #[test]
    fn test_len() {
        let mut m = TreeMap::new();
        assert!(m.insert(3, 6));
        assert!(m.len() == 1);
        assert!(m.insert(0, 0));
        assert!(m.len() == 2);
        assert!(m.insert(4, 8));
        assert!(m.len() == 3);
        assert!(m.remove(&3));
        assert!(m.len() == 2);
        assert!(!m.remove(&5));
        assert!(m.len() == 2);
        assert!(m.insert(2, 4));
        assert!(m.len() == 3);
        assert!(m.insert(1, 2));
        assert!(m.len() == 4);
    }

    #[test]
    fn test_each() {
        let mut m = TreeMap::new();

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
    fn test_each_reverse() {
        let mut m = TreeMap::new();

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
    fn test_eq() {
        let mut a = TreeMap::new();
        let mut b = TreeMap::new();

        assert!(a == b);
        assert!(a.insert(0, 5));
        assert!(a != b);
        assert!(b.insert(0, 4));
        assert!(a != b);
        assert!(a.insert(5, 19));
        assert!(a != b);
        assert!(!b.insert(0, 5));
        assert!(a != b);
        assert!(b.insert(5, 19));
        assert!(a == b);
    }

    #[test]
    fn test_lt() {
        let mut a = TreeMap::new();
        let mut b = TreeMap::new();

        assert!(!(a < b) && !(b < a));
        assert!(b.insert(0, 5));
        assert!(a < b);
        assert!(a.insert(0, 7));
        assert!(!(a < b) && !(b < a));
        assert!(b.insert(-2, 0));
        assert!(b < a);
        assert!(a.insert(-5, 2));
        assert!(a < b);
        assert!(a.insert(6, 2));
        assert!(a < b && !(b < a));
    }

    #[test]
    fn test_ord() {
        let mut a = TreeMap::new();
        let mut b = TreeMap::new();

        assert!(a <= b && a >= b);
        assert!(a.insert(1, 1));
        assert!(a > b && a >= b);
        assert!(b < a && b <= a);
        assert!(b.insert(2, 2));
        assert!(b > a && b >= a);
        assert!(a < b && a <= b);
    }

    #[test]
    fn test_lazy_iterator() {
        let mut m = TreeMap::new();
        let (x1, y1) = (2, 5);
        let (x2, y2) = (9, 12);
        let (x3, y3) = (20, -3);
        let (x4, y4) = (29, 5);
        let (x5, y5) = (103, 3);

        assert!(m.insert(x1, y1));
        assert!(m.insert(x2, y2));
        assert!(m.insert(x3, y3));
        assert!(m.insert(x4, y4));
        assert!(m.insert(x5, y5));

        let m = m;
        let mut a = m.iter();

        assert!(a.next().unwrap() == (&x1, &y1));
        assert!(a.next().unwrap() == (&x2, &y2));
        assert!(a.next().unwrap() == (&x3, &y3));
        assert!(a.next().unwrap() == (&x4, &y4));
        assert!(a.next().unwrap() == (&x5, &y5));

        assert!(a.next().is_none());

        let mut b = m.iter();

        let expected = [(&x1, &y1), (&x2, &y2), (&x3, &y3), (&x4, &y4),
                        (&x5, &y5)];
        let mut i = 0;

        for b.advance |x| {
            assert!(expected[i] == x);
            i += 1;

            if i == 2 {
                break
            }
        }

        for b.advance |x| {
            assert!(expected[i] == x);
            i += 1;
        }
    }
}

#[cfg(test)]
mod test_set {
    use core::iterator::*;
    use super::*;

    #[test]
    fn test_clear() {
        let mut s = TreeSet::new();
        s.clear();
        assert!(s.insert(5));
        assert!(s.insert(12));
        assert!(s.insert(19));
        s.clear();
        assert!(!s.contains(&5));
        assert!(!s.contains(&12));
        assert!(!s.contains(&19));
        assert!(s.is_empty());
    }

    #[test]
    fn test_disjoint() {
        let mut xs = TreeSet::new();
        let mut ys = TreeSet::new();
        assert!(xs.is_disjoint(&ys));
        assert!(ys.is_disjoint(&xs));
        assert!(xs.insert(5));
        assert!(ys.insert(11));
        assert!(xs.is_disjoint(&ys));
        assert!(ys.is_disjoint(&xs));
        assert!(xs.insert(7));
        assert!(xs.insert(19));
        assert!(xs.insert(4));
        assert!(ys.insert(2));
        assert!(ys.insert(-11));
        assert!(xs.is_disjoint(&ys));
        assert!(ys.is_disjoint(&xs));
        assert!(ys.insert(7));
        assert!(!xs.is_disjoint(&ys));
        assert!(!ys.is_disjoint(&xs));
    }

    #[test]
    fn test_subset_and_superset() {
        let mut a = TreeSet::new();
        assert!(a.insert(0));
        assert!(a.insert(5));
        assert!(a.insert(11));
        assert!(a.insert(7));

        let mut b = TreeSet::new();
        assert!(b.insert(0));
        assert!(b.insert(7));
        assert!(b.insert(19));
        assert!(b.insert(250));
        assert!(b.insert(11));
        assert!(b.insert(200));

        assert!(!a.is_subset(&b));
        assert!(!a.is_superset(&b));
        assert!(!b.is_subset(&a));
        assert!(!b.is_superset(&a));

        assert!(b.insert(5));

        assert!(a.is_subset(&b));
        assert!(!a.is_superset(&b));
        assert!(!b.is_subset(&a));
        assert!(b.is_superset(&a));
    }

    #[test]
    fn test_each() {
        let mut m = TreeSet::new();

        assert!(m.insert(3));
        assert!(m.insert(0));
        assert!(m.insert(4));
        assert!(m.insert(2));
        assert!(m.insert(1));

        let mut n = 0;
        for m.each |x| {
            assert!(*x == n);
            n += 1
        }
    }

    #[test]
    fn test_each_reverse() {
        let mut m = TreeSet::new();

        assert!(m.insert(3));
        assert!(m.insert(0));
        assert!(m.insert(4));
        assert!(m.insert(2));
        assert!(m.insert(1));

        let mut n = 4;
        for m.each_reverse |x| {
            assert!(*x == n);
            n -= 1
        }
    }

    fn check(a: &[int], b: &[int], expected: &[int],
             f: &fn(&TreeSet<int>, &TreeSet<int>, f: &fn(&int) -> bool)) {
        let mut set_a = TreeSet::new();
        let mut set_b = TreeSet::new();

        for a.each |x| { assert!(set_a.insert(*x)) }
        for b.each |y| { assert!(set_b.insert(*y)) }

        let mut i = 0;
        for f(&set_a, &set_b) |x| {
            assert!(*x == expected[i]);
            i += 1;
        }
        assert!(i == expected.len());
    }

    #[test]
    fn test_intersection() {
        fn check_intersection(a: &[int], b: &[int], expected: &[int]) {
            check(a, b, expected, |x, y, z| x.intersection(y, z))
        }

        check_intersection([], [], []);
        check_intersection([1, 2, 3], [], []);
        check_intersection([], [1, 2, 3], []);
        check_intersection([2], [1, 2, 3], [2]);
        check_intersection([1, 2, 3], [2], [2]);
        check_intersection([11, 1, 3, 77, 103, 5, -5],
                           [2, 11, 77, -9, -42, 5, 3],
                           [3, 5, 11, 77]);
    }

    #[test]
    fn test_difference() {
        fn check_difference(a: &[int], b: &[int], expected: &[int]) {
            check(a, b, expected, |x, y, z| x.difference(y, z))
        }

        check_difference([], [], []);
        check_difference([1, 12], [], [1, 12]);
        check_difference([], [1, 2, 3, 9], []);
        check_difference([1, 3, 5, 9, 11],
                         [3, 9],
                         [1, 5, 11]);
        check_difference([-5, 11, 22, 33, 40, 42],
                         [-12, -5, 14, 23, 34, 38, 39, 50],
                         [11, 22, 33, 40, 42]);
    }

    #[test]
    fn test_symmetric_difference() {
        fn check_symmetric_difference(a: &[int], b: &[int],
                                      expected: &[int]) {
            check(a, b, expected, |x, y, z| x.symmetric_difference(y, z))
        }

        check_symmetric_difference([], [], []);
        check_symmetric_difference([1, 2, 3], [2], [1, 3]);
        check_symmetric_difference([2], [1, 2, 3], [1, 3]);
        check_symmetric_difference([1, 3, 5, 9, 11],
                                   [-2, 3, 9, 14, 22],
                                   [-2, 1, 5, 11, 14, 22]);
    }

    #[test]
    fn test_union() {
        fn check_union(a: &[int], b: &[int],
                                      expected: &[int]) {
            check(a, b, expected, |x, y, z| x.union(y, z))
        }

        check_union([], [], []);
        check_union([1, 2, 3], [2], [1, 2, 3]);
        check_union([2], [1, 2, 3], [1, 2, 3]);
        check_union([1, 3, 5, 9, 11, 16, 19, 24],
                    [-2, 1, 5, 9, 13, 19],
                    [-2, 1, 3, 5, 9, 11, 13, 16, 19, 24]);
    }

    #[test]
    fn test_zip() {
        let mut x = TreeSet::new();
        x.insert(5u);
        x.insert(12u);
        x.insert(11u);

        let mut y = TreeSet::new();
        y.insert("foo");
        y.insert("bar");

        let x = x;
        let y = y;
        let mut z = x.iter().zip(y.iter());

        // FIXME: #5801: this needs a type hint to compile...
        let result: Option<(&uint, & &'static str)> = z.next();
        assert!(result.unwrap() == (&5u, & &"bar"));

        let result: Option<(&uint, & &'static str)> = z.next();
        assert!(result.unwrap() == (&11u, & &"foo"));

        let result: Option<(&uint, & &'static str)> = z.next();
        assert!(result.is_none());
    }

    #[test]
    fn test_swap() {
        let mut m = TreeMap::new();
        assert!(m.swap(1, 2) == None);
        assert!(m.swap(1, 3) == Some(2));
        assert!(m.swap(1, 4) == Some(3));
    }

    #[test]
    fn test_pop() {
        let mut m = TreeMap::new();
        m.insert(1, 2);
        assert!(m.pop(&1) == Some(2));
        assert!(m.pop(&1) == None);
    }
}
