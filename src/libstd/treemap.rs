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
//! `Ord`, and that the `lt` method provides a total ordering.

#[forbid(deprecated_mode)];

use core::container::{Container, Mutable, Map, Set};
use core::cmp::{Eq, Ord};
use core::option::{Option, Some, None};
use core::prelude::*;

// This is implemented as an AA tree, which is a simplified variation of
// a red-black tree where where red (horizontal) nodes can only be added
// as a right child. The time complexity is the same, and re-balancing
// operations are more frequent but also cheaper.

// Future improvements:

// range search - O(log n) retrieval of an iterator from some key

// implement Ord for TreeSet
// could be superset/subset-based or in-order lexicographic comparison... but
// there are methods for is_superset/is_subset so lexicographic is more useful

// (possibly) implement the overloads Python does for sets:
//   * union: |
//   * intersection: &
//   * difference: -
//   * symmetric difference: ^
// These would be convenient since the methods work like `each`

pub struct TreeMap<K: Ord, V> {
    priv root: Option<~TreeNode<K, V>>,
    priv length: uint
}

impl <K: Eq Ord, V: Eq> TreeMap<K, V>: Eq {
    pure fn eq(&self, other: &TreeMap<K, V>) -> bool {
        if self.len() != other.len() {
            false
        } else {
            let mut x = self.iter();
            let mut y = other.iter();
            for self.len().times {
                unsafe { // unsafe as a purity workaround
                    // ICE: x.next() != y.next()

                    x = x.next();
                    y = y.next();
                    let (x1, x2) = x.get().unwrap();
                    let (y1, y2) = y.get().unwrap();

                    if x1 != y1 || x2 != y2 {
                        return false
                    }
                }
            }
            true
        }
    }
    pure fn ne(&self, other: &TreeMap<K, V>) -> bool { !self.eq(other) }
}

impl <K: Ord, V> TreeMap<K, V>: Container {
    /// Return the number of elements in the map
    pure fn len(&self) -> uint { self.length }

    /// Return true if the map contains no elements
    pure fn is_empty(&self) -> bool { self.root.is_none() }
}

impl <K: Ord, V> TreeMap<K, V>: Mutable {
    /// Clear the map, removing all key-value pairs.
    fn clear(&mut self) {
        self.root = None;
        self.length = 0
    }
}

impl <K: Ord, V> TreeMap<K, V>: Map<K, V> {
    /// Return true if the map contains a value for the specified key
    pure fn contains_key(&self, key: &K) -> bool {
        self.find(key).is_some()
    }

    /// Visit all key-value pairs in order
    pure fn each(&self, f: fn(&K, &V) -> bool) { each(&self.root, f) }

    /// Visit all keys in order
    pure fn each_key(&self, f: fn(&K) -> bool) { self.each(|k, _| f(k)) }

    /// Visit all values in order
    pure fn each_value(&self, f: fn(&V) -> bool) { self.each(|_, v| f(v)) }

    /// Return the value corresponding to the key in the map
    pure fn find(&self, key: &K) -> Option<&self/V> {
        let mut current: &self/Option<~TreeNode<K, V>> = &self.root;
        loop {
            match *current {
              Some(ref r) => {
                let r: &self/~TreeNode<K, V> = r; // FIXME: #3148
                if *key < r.key {
                    current = &r.left;
                } else if r.key < *key {
                    current = &r.right;
                } else {
                    return Some(&r.value);
                }
              }
              None => return None
            }
        }
    }

    /// Insert a key-value pair into the map. An existing value for a
    /// key is replaced by the new value. Return true if the key did
    /// not already exist in the map.
    fn insert(&mut self, key: K, value: V) -> bool {
        let ret = insert(&mut self.root, key, value);
        if ret { self.length += 1 }
        ret
    }

    /// Remove a key-value pair from the map. Return true if the key
    /// was present in the map, otherwise false.
    fn remove(&mut self, key: &K) -> bool {
        let ret = remove(&mut self.root, key);
        if ret { self.length -= 1 }
        ret
    }
}

impl <K: Ord, V> TreeMap<K, V> {
    /// Create an empty TreeMap
    static pure fn new() -> TreeMap<K, V> { TreeMap{root: None, length: 0} }

    /// Visit all key-value pairs in reverse order
    pure fn each_reverse(&self, f: fn(&K, &V) -> bool) {
        each_reverse(&self.root, f);
    }

    /// Visit all keys in reverse order
    pure fn each_key_reverse(&self, f: fn(&K) -> bool) {
        self.each_reverse(|k, _| f(k))
    }

    /// Visit all values in reverse order
    pure fn each_value_reverse(&self, f: fn(&V) -> bool) {
        self.each_reverse(|_, v| f(v))
    }

    /// Get a lazy iterator over the key-value pairs in the map.
    /// Requires that it be frozen (immutable).
    pure fn iter(&self) -> TreeMapIterator/&self<K, V> {
        TreeMapIterator{stack: ~[], node: &self.root, current: None}
    }
}

/// Lazy forward iterator over a map
pub struct TreeMapIterator<K: Ord, V> {
    priv stack: ~[&~TreeNode<K, V>],
    priv node: &Option<~TreeNode<K, V>>,
    priv current: Option<&~TreeNode<K, V>>
}

impl <K: Ord, V> TreeMapIterator<K, V> {
    // Returns the current node, or None if this iterator is at the end.
    fn get(&const self) -> Option<(&self/K, &self/V)> {
        match self.current {
            Some(res) => Some((&res.key, &res.value)),
            None => None
        }
    }

    /// Advance the iterator to the next node (in order). If this iterator
    /// is finished, does nothing.
    fn next(self) -> TreeMapIterator/&self<K, V> {
        let mut this = self;
        while !this.stack.is_empty() || this.node.is_some() {
            match *this.node {
              Some(ref x) => {
                this.stack.push(x);
                this.node = &x.left;
              }
              None => {
                let res = this.stack.pop();
                this.node = &res.right;
                this.current = Some(res);
                return this;
              }
            }
        }
        this.current = None;
        return this;
    }
}

pub struct TreeSet<T: Ord> {
    priv map: TreeMap<T, ()>
}

impl <T: Ord> TreeSet<T>: iter::BaseIter<T> {
    /// Visit all values in order
    pure fn each(&self, f: fn(&T) -> bool) { self.map.each_key(f) }
    pure fn size_hint(&self) -> Option<uint> { Some(self.len()) }
}

impl <T: Eq Ord> TreeSet<T>: Eq {
    pure fn eq(&self, other: &TreeSet<T>) -> bool { self.map == other.map }
    pure fn ne(&self, other: &TreeSet<T>) -> bool { self.map != other.map }
}

impl <T: Ord> TreeSet<T>: Container {
    /// Return the number of elements in the set
    pure fn len(&self) -> uint { self.map.len() }

    /// Return true if the set contains no elements
    pure fn is_empty(&self) -> bool { self.map.is_empty() }
}

impl <T: Ord> TreeSet<T>: Mutable {
    /// Clear the set, removing all values.
    fn clear(&mut self) { self.map.clear() }
}

impl <T: Ord> TreeSet<T>: Set<T> {
    /// Return true if the set contains a value
    pure fn contains(&self, value: &T) -> bool {
        self.map.contains_key(value)
    }

    /// Add a value to the set. Return true if the value was not already
    /// present in the set.
    fn insert(&mut self, value: T) -> bool { self.map.insert(value, ()) }

    /// Remove a value from the set. Return true if the value was
    /// present in the set.
    fn remove(&mut self, value: &T) -> bool { self.map.remove(value) }
}

impl <T: Ord> TreeSet<T> {
    /// Create an empty TreeSet
    static pure fn new() -> TreeSet<T> { TreeSet{map: TreeMap::new()} }

    /// Visit all values in reverse order
    pure fn each_reverse(&self, f: fn(&T) -> bool) {
        self.map.each_key_reverse(f)
    }

    /// Get a lazy iterator over the values in the set.
    /// Requires that it be frozen (immutable).
    pure fn iter(&self) -> TreeSetIterator/&self<T> {
        TreeSetIterator{iter: self.map.iter()}
    }

    /// Return true if the set has no elements in common with `other`.
    /// This is equivalent to checking for an empty intersection.
    pure fn is_disjoint(&self, other: &TreeSet<T>) -> bool {
        let mut x = self.iter();
        let mut y = other.iter();
        unsafe { // purity workaround
            x = x.next();
            y = y.next();
            let mut a = x.get();
            let mut b = y.get();
            while a.is_some() && b.is_some() {
                let a1 = a.unwrap();
                let b1 = b.unwrap();
                if a1 < b1 {
                    x = x.next();
                    a = x.get();
                } else if b1 < a1 {
                    y = y.next();
                    b = y.get();
                } else {
                    return false;
                }
            }
        }
        true
    }

    /// Check of the set is a subset of another
    pure fn is_subset(&self, other: &TreeSet<T>) -> bool {
        other.is_superset(self)
    }

    /// Check of the set is a superset of another
    pure fn is_superset(&self, other: &TreeSet<T>) -> bool {
        let mut x = self.iter();
        let mut y = other.iter();
        unsafe { // purity workaround
            x = x.next();
            y = y.next();
            let mut a = x.get();
            let mut b = y.get();
            while b.is_some() {
                if a.is_none() {
                    return false
                }

                let a1 = a.unwrap();
                let b1 = b.unwrap();

                if b1 < a1 {
                    return false
                }

                if !(a1 < b1) {
                    y = y.next();
                    b = y.get();
                }
                x = x.next();
                a = x.get();
            }
        }
        true
    }

    /// Visit the values (in-order) representing the difference
    pure fn difference(&self, other: &TreeSet<T>, f: fn(&T) -> bool) {
        let mut x = self.iter();
        let mut y = other.iter();

        unsafe { // purity workaround
            x = x.next();
            y = y.next();
            let mut a = x.get();
            let mut b = y.get();

            while a.is_some() {
                if b.is_none() {
                    return do a.while_some() |a1| {
                        if f(a1) { x = x.next(); x.get() } else { None }
                    }
                }

                let a1 = a.unwrap();
                let b1 = b.unwrap();

                if a1 < b1 {
                    if !f(a1) { return }
                    x = x.next();
                    a = x.get();
                } else {
                    if !(b1 < a1) { x = x.next(); a = x.get() }
                    y = y.next();
                    b = y.get();
                }
            }
        }
    }

    /// Visit the values (in-order) representing the symmetric difference
    pure fn symmetric_difference(&self, other: &TreeSet<T>,
                                 f: fn(&T) -> bool) {
        let mut x = self.iter();
        let mut y = other.iter();

        unsafe { // purity workaround
            x = x.next();
            y = y.next();
            let mut a = x.get();
            let mut b = y.get();

            while a.is_some() {
                if b.is_none() {
                    return do a.while_some() |a1| {
                        if f(a1) { x.next(); x.get() } else { None }
                    }
                }

                let a1 = a.unwrap();
                let b1 = b.unwrap();

                if a1 < b1 {
                    if !f(a1) { return }
                    x = x.next();
                    a = x.get();
                } else {
                    if b1 < a1 {
                        if !f(b1) { return }
                    } else {
                        x = x.next();
                        a = x.get();
                    }
                    y = y.next();
                    b = y.get();
                }
            }
            do b.while_some |b1| {
                if f(b1) { y = y.next(); y.get() } else { None }
            }
        }
    }

    /// Visit the values (in-order) representing the intersection
    pure fn intersection(&self, other: &TreeSet<T>, f: fn(&T) -> bool) {
        let mut x = self.iter();
        let mut y = other.iter();

        unsafe { // purity workaround
            x = x.next();
            y = y.next();
            let mut a = x.get();
            let mut b = y.get();

            while a.is_some() && b.is_some() {
                let a1 = a.unwrap();
                let b1 = b.unwrap();
                if a1 < b1 {
                    x = x.next();
                    a = x.get();
                } else {
                    if !(b1 < a1) {
                        if !f(a1) { return }
                    }
                    y = y.next();
                    b = y.get();
                }
            }
        }
    }

    /// Visit the values (in-order) representing the union
    pure fn union(&self, other: &TreeSet<T>, f: fn(&T) -> bool) {
        let mut x = self.iter();
        let mut y = other.iter();

        unsafe { // purity workaround
            x = x.next();
            y = y.next();
            let mut a = x.get();
            let mut b = y.get();

            while a.is_some() {
                if b.is_none() {
                    return do a.while_some() |a1| {
                        if f(a1) { x = x.next(); x.get() } else { None }
                    }
                }

                let a1 = a.unwrap();
                let b1 = b.unwrap();

                if b1 < a1 {
                    if !f(b1) { return }
                    y = y.next();
                    b = y.get();
                } else {
                    if !f(a1) { return }
                    if !(a1 < b1) {
                        y = y.next();
                        b = y.get()
                    }
                    x = x.next();
                    a = x.get();
                }
            }
        }
    }
}

/// Lazy forward iterator over a set
pub struct TreeSetIterator<T: Ord> {
    priv iter: TreeMapIterator<T, ()>
}

impl <T: Ord> TreeSetIterator<T> {
    /// Returns the current node, or None if this iterator is at the end.
    fn get(&const self) -> Option<&self/T> {
        match self.iter.get() {
            None => None,
            Some((k, _)) => Some(k)
        }
    }

    /// Advance the iterator to the next node (in order). If this iterator is
    /// finished, does nothing.
    fn next(self) -> TreeSetIterator/&self<T> {
        TreeSetIterator { iter: self.iter.next() }
    }
}

// Nodes keep track of their level in the tree, starting at 1 in the
// leaves and with a red child sharing the level of the parent.
struct TreeNode<K: Ord, V> {
    key: K,
    value: V,
    left: Option<~TreeNode<K, V>>,
    right: Option<~TreeNode<K, V>>,
    level: uint
}

impl <K: Ord, V> TreeNode<K, V> {
    #[inline(always)]
    static pure fn new(key: K, value: V) -> TreeNode<K, V> {
        TreeNode{key: key, value: value, left: None, right: None, level: 1}
    }
}

pure fn each<K: Ord, V>(node: &Option<~TreeNode<K, V>>,
                        f: fn(&K, &V) -> bool) {
    do node.map |x| {
        each(&x.left, f);
        if f(&x.key, &x.value) { each(&x.right, f) }
    };
}

pure fn each_reverse<K: Ord, V>(node: &Option<~TreeNode<K, V>>,
                                f: fn(&K, &V) -> bool) {
    do node.map |x| {
        each_reverse(&x.right, f);
        if f(&x.key, &x.value) { each_reverse(&x.left, f) }
    };
}

// Remove left horizontal link by rotating right
fn skew<K: Ord, V>(node: ~TreeNode<K, V>) -> ~TreeNode<K, V> {
    if node.left.map_default(false, |x| x.level == node.level) {
        let mut node = node;
        let mut save = node.left.swap_unwrap();
        node.left <-> save.right; // save.right now None
        save.right = Some(node);
        save
    } else {
        node // nothing to do
    }
}

// Remove dual horizontal link by rotating left and increasing level of
// the parent
fn split<K: Ord, V>(node: ~TreeNode<K, V>) -> ~TreeNode<K, V> {
    if node.right.map_default(false,
      |x| x.right.map_default(false, |y| y.level == node.level)) {
        let mut node = node;
        let mut save = node.right.swap_unwrap();
        node.right <-> save.left; // save.left now None
        save.left = Some(node);
        save.level += 1;
        save
    } else {
        node // nothing to do
    }
}

fn insert<K: Ord, V>(node: &mut Option<~TreeNode<K, V>>, key: K,
                     value: V) -> bool {
    if node.is_none() {
        *node = Some(~TreeNode::new(key, value));
        true
    } else {
        let mut save = node.swap_unwrap();
        if key < save.key {
            let inserted = insert(&mut save.left, key, value);
            *node = Some(split(skew(save))); // re-balance, if necessary
            inserted
        } else if save.key < key {
            let inserted = insert(&mut save.right, key, value);
            *node = Some(split(skew(save))); // re-balance, if necessary
            inserted
        } else {
            save.key = key;
            save.value = value;
            *node = Some(save);
            false
        }
    }
}

fn remove<K: Ord, V>(node: &mut Option<~TreeNode<K, V>>, key: &K) -> bool {
    fn heir_swap<K: Ord, V>(node: &mut TreeNode<K, V>,
                            child: &mut Option<~TreeNode<K, V>>) {
        // *could* be done without recursion, but it won't borrow check
        do child.mutate |child| {
            let mut child = child;
            if child.right.is_some() {
                heir_swap(&mut *node, &mut child.right);
            } else {
                node.key <-> child.key;
                node.value <-> child.value;
            }
            child
        }
    }

    if node.is_none() {
        return false // bottom of tree
    } else {
        let mut save = node.swap_unwrap();

        let removed = if save.key < *key {
            remove(&mut save.right, key)
        } else if *key < save.key {
            remove(&mut save.left, key)
        } else {
            if save.left.is_some() {
                if save.right.is_some() {
                    let mut left = save.left.swap_unwrap();
                    if left.right.is_some() {
                        heir_swap(save, &mut left.right);
                        save.left = Some(left);
                        remove(&mut save.left, key);
                    } else {
                        save.key <-> left.key;
                        save.value <-> left.value;
                        save.left = Some(left);
                        remove(&mut save.left, key);
                    }
                } else {
                    save = save.left.swap_unwrap();
                }
            } else if save.right.is_some() {
                save = save.right.swap_unwrap();
            } else {
                return true // leaf
            }
            true
        };

        let left_level = save.left.map_default(0, |x| x.level);
        let right_level = save.right.map_default(0, |x| x.level);

        // re-balance, if necessary
        if left_level < save.level - 1 || right_level < save.level - 1 {
            save.level -= 1;

            if right_level > save.level {
                do save.right.mutate |x| {
                    let mut x = x; x.level = save.level; x
                }
            }

            save = skew(save);

            do save.right.mutate |right| {
                let mut right = skew(right);
                right.right.mutate(skew);
                right
            }
            save = split(save);
            save.right.mutate(split);
        }

        *node = Some(save);
        removed
    }
}

#[cfg(test)]
mod test_treemap {
    use super::*;
    use core::str;

    #[test]
    fn find_empty() {
        let m = TreeMap::new::<int, int>(); assert m.find(&5) == None;
    }

    #[test]
    fn find_not_found() {
        let mut m = TreeMap::new();
        assert m.insert(1, 2);
        assert m.insert(5, 3);
        assert m.insert(9, 3);
        assert m.find(&2) == None;
    }

    #[test]
    fn insert_replace() {
        let mut m = TreeMap::new();
        assert m.insert(5, 2);
        assert m.insert(2, 9);
        assert !m.insert(2, 11);
        assert m.find(&2).unwrap() == &11;
    }

    #[test]
    fn test_clear() {
        let mut m = TreeMap::new();
        m.clear();
        assert m.insert(5, 11);
        assert m.insert(12, -3);
        assert m.insert(19, 2);
        m.clear();
        assert m.find(&5).is_none();
        assert m.find(&12).is_none();
        assert m.find(&19).is_none();
        assert m.is_empty();
    }

    #[test]
    fn u8_map() {
        let mut m = TreeMap::new();

        let k1 = str::to_bytes(~"foo");
        let k2 = str::to_bytes(~"bar");
        let v1 = str::to_bytes(~"baz");
        let v2 = str::to_bytes(~"foobar");

        m.insert(k1, v1);
        m.insert(k2, v2);

        assert m.find(&k2) == Some(&v2);
        assert m.find(&k1) == Some(&v1);
    }

    fn check_equal<K: Eq Ord, V: Eq>(ctrl: &[(K, V)], map: &TreeMap<K, V>) {
        assert ctrl.is_empty() == map.is_empty();
        for ctrl.each |x| {
            let &(k, v) = x;
            assert map.find(&k).unwrap() == &v
        }
        for map.each |map_k, map_v| {
            let mut found = false;
            for ctrl.each |x| {
                let &(ctrl_k, ctrl_v) = x;
                if *map_k == ctrl_k {
                    assert *map_v == ctrl_v;
                    found = true;
                    break;
                }
            }
            assert found;
        }
    }

    fn check_left<K: Ord, V>(node: &Option<~TreeNode<K, V>>,
                             parent: &~TreeNode<K, V>) {
        match *node {
          Some(ref r) => {
            assert r.key < parent.key;
            assert r.level == parent.level - 1; // left is black
            check_left(&r.left, r);
            check_right(&r.right, r, false);
          }
          None => assert parent.level == 1 // parent is leaf
        }
    }

    fn check_right<K: Ord, V>(node: &Option<~TreeNode<K, V>>,
                              parent: &~TreeNode<K, V>, parent_red: bool) {
        match *node {
          Some(ref r) => {
            assert r.key > parent.key;
            let red = r.level == parent.level;
            if parent_red { assert !red } // no dual horizontal links
            assert red || r.level == parent.level - 1; // right red or black
            check_left(&r.left, r);
            check_right(&r.right, r, red);
          }
          None => assert parent.level == 1 // parent is leaf
        }
    }

    fn check_structure<K: Ord, V>(map: &TreeMap<K, V>) {
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
        assert map.find(&5).is_none();

        let rng = rand::seeded_rng(&~[42]);

        for 3.times {
            for 90.times {
                let k = rng.gen_int();
                let v = rng.gen_int();
                if !ctrl.contains(&(k, v)) {
                    assert map.insert(k, v);
                    ctrl.push((k, v));
                    check_structure(&map);
                    check_equal(ctrl, &map);
                }
            }

            for 30.times {
                let r = rng.gen_uint_range(0, ctrl.len());
                let (key, _) = vec::remove(&mut ctrl, r);
                assert map.remove(&key);
                check_structure(&map);
                check_equal(ctrl, &map);
            }
        }
    }

    #[test]
    fn test_len() {
        let mut m = TreeMap::new();
        assert m.insert(3, 6);
        assert m.len() == 1;
        assert m.insert(0, 0);
        assert m.len() == 2;
        assert m.insert(4, 8);
        assert m.len() == 3;
        assert m.remove(&3);
        assert m.len() == 2;
        assert !m.remove(&5);
        assert m.len() == 2;
        assert m.insert(2, 4);
        assert m.len() == 3;
        assert m.insert(1, 2);
        assert m.len() == 4;
    }

    #[test]
    fn test_each() {
        let mut m = TreeMap::new();

        assert m.insert(3, 6);
        assert m.insert(0, 0);
        assert m.insert(4, 8);
        assert m.insert(2, 4);
        assert m.insert(1, 2);

        let mut n = 0;
        for m.each |k, v| {
            assert *k == n;
            assert *v == n * 2;
            n += 1;
        }
    }

    #[test]
    fn test_each_reverse() {
        let mut m = TreeMap::new();

        assert m.insert(3, 6);
        assert m.insert(0, 0);
        assert m.insert(4, 8);
        assert m.insert(2, 4);
        assert m.insert(1, 2);

        let mut n = 4;
        for m.each_reverse |k, v| {
            assert *k == n;
            assert *v == n * 2;
            n -= 1;
        }
    }

    #[test]
    fn test_eq() {
        let mut a = TreeMap::new();
        let mut b = TreeMap::new();

        assert a == b;
        assert a.insert(0, 5);
        assert a != b;
        assert b.insert(0, 4);
        assert a != b;
        assert a.insert(5, 19);
        assert a != b;
        assert !b.insert(0, 5);
        assert a != b;
        assert b.insert(5, 19);
        assert a == b;
    }

    #[test]
    fn test_lazy_iterator() {
        let mut m = TreeMap::new();
        let (x1, y1) = (2, 5);
        let (x2, y2) = (9, 12);
        let (x3, y3) = (20, -3);
        let (x4, y4) = (29, 5);
        let (x5, y5) = (103, 3);

        assert m.insert(x1, y1);
        assert m.insert(x2, y2);
        assert m.insert(x3, y3);
        assert m.insert(x4, y4);
        assert m.insert(x5, y5);

        let m = m;
        let mut iter = m.iter();

        // ICE:
        //assert iter.next() == Some((&x1, &y1));
        //assert iter.next().eq(&Some((&x1, &y1)));

        iter = iter.next();
        assert iter.get().unwrap() == (&x1, &y1);
        iter = iter.next();
        assert iter.get().unwrap() == (&x2, &y2);
        iter = iter.next();
        assert iter.get().unwrap() == (&x3, &y3);
        iter = iter.next();
        assert iter.get().unwrap() == (&x4, &y4);
        iter = iter.next();
        assert iter.get().unwrap() == (&x5, &y5);

        // ICE:
        //assert iter.next() == None;
        //assert iter.next().eq(&None);

        iter = iter.next();
        assert iter.get().is_none();
    }
}

#[cfg(test)]
mod test_set {
    use super::*;

    #[test]
    fn test_clear() {
        let mut s = TreeSet::new();
        s.clear();
        assert s.insert(5);
        assert s.insert(12);
        assert s.insert(19);
        s.clear();
        assert !s.contains(&5);
        assert !s.contains(&12);
        assert !s.contains(&19);
        assert s.is_empty();
    }

    #[test]
    fn test_disjoint() {
        let mut xs = TreeSet::new();
        let mut ys = TreeSet::new();
        assert xs.is_disjoint(&ys);
        assert ys.is_disjoint(&xs);
        assert xs.insert(5);
        assert ys.insert(11);
        assert xs.is_disjoint(&ys);
        assert ys.is_disjoint(&xs);
        assert xs.insert(7);
        assert xs.insert(19);
        assert xs.insert(4);
        assert ys.insert(2);
        assert ys.insert(-11);
        assert xs.is_disjoint(&ys);
        assert ys.is_disjoint(&xs);
        assert ys.insert(7);
        assert !xs.is_disjoint(&ys);
        assert !ys.is_disjoint(&xs);
    }

    #[test]
    fn test_subset_and_superset() {
        let mut a = TreeSet::new();
        assert a.insert(0);
        assert a.insert(5);
        assert a.insert(11);
        assert a.insert(7);

        let mut b = TreeSet::new();
        assert b.insert(0);
        assert b.insert(7);
        assert b.insert(19);
        assert b.insert(250);
        assert b.insert(11);
        assert b.insert(200);

        assert !a.is_subset(&b);
        assert !a.is_superset(&b);
        assert !b.is_subset(&a);
        assert !b.is_superset(&a);

        assert b.insert(5);

        assert a.is_subset(&b);
        assert !a.is_superset(&b);
        assert !b.is_subset(&a);
        assert b.is_superset(&a);
    }

    #[test]
    fn test_each() {
        let mut m = TreeSet::new();

        assert m.insert(3);
        assert m.insert(0);
        assert m.insert(4);
        assert m.insert(2);
        assert m.insert(1);

        let mut n = 0;
        for m.each |x| {
            assert *x == n;
            n += 1
        }
    }

    #[test]
    fn test_each_reverse() {
        let mut m = TreeSet::new();

        assert m.insert(3);
        assert m.insert(0);
        assert m.insert(4);
        assert m.insert(2);
        assert m.insert(1);

        let mut n = 4;
        for m.each_reverse |x| {
            assert *x == n;
            n -= 1
        }
    }

    #[test]
    fn test_intersection() {
        let mut a = TreeSet::new();
        let mut b = TreeSet::new();

        assert a.insert(11);
        assert a.insert(1);
        assert a.insert(3);
        assert a.insert(77);
        assert a.insert(103);
        assert a.insert(5);
        assert a.insert(-5);

        assert b.insert(2);
        assert b.insert(11);
        assert b.insert(77);
        assert b.insert(-9);
        assert b.insert(-42);
        assert b.insert(5);
        assert b.insert(3);

        let mut i = 0;
        let expected = [3, 5, 11, 77];
        for a.intersection(&b) |x| {
            assert *x == expected[i];
            i += 1
        }
        assert i == expected.len();
    }

    #[test]
    fn test_difference() {
        let mut a = TreeSet::new();
        let mut b = TreeSet::new();

        assert a.insert(1);
        assert a.insert(3);
        assert a.insert(5);
        assert a.insert(9);
        assert a.insert(11);

        assert b.insert(3);
        assert b.insert(9);

        let mut i = 0;
        let expected = [1, 5, 11];
        for a.difference(&b) |x| {
            assert *x == expected[i];
            i += 1
        }
        assert i == expected.len();
    }

    #[test]
    fn test_symmetric_difference() {
        let mut a = TreeSet::new();
        let mut b = TreeSet::new();

        assert a.insert(1);
        assert a.insert(3);
        assert a.insert(5);
        assert a.insert(9);
        assert a.insert(11);

        assert b.insert(-2);
        assert b.insert(3);
        assert b.insert(9);
        assert b.insert(14);
        assert b.insert(22);

        let mut i = 0;
        let expected = [-2, 1, 5, 11, 14, 22];
        for a.symmetric_difference(&b) |x| {
            assert *x == expected[i];
            i += 1
        }
        assert i == expected.len();
    }

    #[test]
    fn test_union() {
        let mut a = TreeSet::new();
        let mut b = TreeSet::new();

        assert a.insert(1);
        assert a.insert(3);
        assert a.insert(5);
        assert a.insert(9);
        assert a.insert(11);
        assert a.insert(16);
        assert a.insert(19);
        assert a.insert(24);

        assert b.insert(-2);
        assert b.insert(1);
        assert b.insert(5);
        assert b.insert(9);
        assert b.insert(13);
        assert b.insert(19);

        let mut i = 0;
        let expected = [-2, 1, 3, 5, 9, 11, 13, 16, 19, 24];
        for a.union(&b) |x| {
            assert *x == expected[i];
            i += 1
        }
        assert i == expected.len();
    }
}
