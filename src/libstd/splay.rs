// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Contains an implementation of splay trees where each node has a key/value
//! pair to be used in maps and sets. The only requirement is that the key must
//! implement the TotalOrd trait.

use core::iterator::*;
use core::util;

/// The implementation of this splay tree is largely based on the java code at:
/// https://github.com/cpdomina/SplaySplayMap. This version of splaying is a
/// top-down splay operation.
pub struct SplayMap<K, V> {
    priv root: Option<~Node<K, V>>,
    priv size: uint,
}

pub struct SplaySet<T> {
    priv map: SplayMap<T, ()>
}

struct Node<K, V> {
  key: K,
  value: V,
  left: Option<~Node<K, V>>,
  right: Option<~Node<K, V>>,
}

pub impl<K: TotalOrd, V> SplayMap<K, V> {
    fn new() -> SplayMap<K, V> {
        SplayMap{ root: None, size: 0 }
    }

    /// Performs a splay operation on the tree, moving a key to the root, or one
    /// of the closest values to the key to the root.
    fn splay(&mut self, key: &K) {
        let mut root = util::replace(&mut self.root, None).unwrap();
        let mut newleft = None;
        let mut newright = None;

        // Yes, these are backwards, that's intentional.
        root = root.splay(key, &mut newright, &mut newleft);

        root.left = newright;
        root.right = newleft;
        self.root = Some(root);
    }

    /// Get a lazy iterator over the key-value pairs in the map.
    /// Requires that it be frozen (immutable).
    fn iter<'a>(&'a self) -> BSTIterator<'a, Node<K, V>, (&'a K, &'a V)> {
        BSTIterator::new(&self.root,
                         |n| (&n.key, &n.value),
                         |n| &n.left,
                         |n| &n.right)
    }
}

impl<K, V> Container for SplayMap<K, V> {
    fn len(&const self) -> uint { self.size }
    fn is_empty(&const self) -> bool { self.len() == 0 }
}

impl<K, V> Mutable for SplayMap<K, V> {
    fn clear(&mut self) {
        self.root = None;
        self.size = 0;
    }
}

impl<K: TotalOrd, V> Map<K, V> for SplayMap<K, V> {
    /// Return true if the map contains a value for the specified key
    #[inline(always)]
    fn contains_key(&self, key: &K) -> bool {
        self.find(key).is_some()
    }

    /// Visit all values in order
    #[inline(always)]
    fn each<'a>(&'a self, f: &fn(&K, &'a V) -> bool) {
        for self.root.each |n| {
            n.each(f);
        }
    }

    /// Iterate over the map and mutate the contained values
    #[inline(always)]
    fn mutate_values(&mut self, f: &fn(&K, &mut V) -> bool) {
        match self.root {
            None => {}
            Some(ref mut node) => node.each_mut(f)
        }
    }

    /// Visit all keys in order
    #[inline(always)]
    fn each_key(&self, f: &fn(&K) -> bool) { self.each(|k, _| f(k)) }

    /// Visit all values in order
    #[inline(always)]
    fn each_value<'a>(&'a self, f: &fn(&'a V) -> bool) {
        self.each(|_, v| f(v))
    }

    /// Insert a key-value pair into the map. An existing value for a
    /// key is replaced by the new value. Return true if the key did
    /// not already exist in the map.
    #[inline(always)]
    fn insert(&mut self, key: K, value: V) -> bool {
        self.swap(key, value).is_none()
    }

    /// Insert a key-value pair from the map. If the key already had a value
    /// present in the map, that value is returned. Otherwise None is returned.
    fn swap(&mut self, key: K, value: V) -> Option<V> {
        if self.root.is_none() {
            self.root = Some(Node::new(key, value));
            self.size += 1;
            return None;
        }

        self.splay(&key);
        let mut root = util::replace(&mut self.root, None).unwrap();

        match key.cmp(&root.key) {
            Equal => {
                let ret = Some(util::replace(&mut root.value, value));
                self.root = Some(root);
                return ret;
            }
            Less => {
                let mut me = Node::new(key, value);
                me.left = util::replace(&mut root.left, None);
                me.right = Some(root);
                self.root = Some(me);
            }
            Greater => {
                let mut me = Node::new(key, value);
                me.right = util::replace(&mut root.right, None);
                me.left = Some(root);
                self.root = Some(me);
            }
        }

        self.size += 1;
        return None;
    }

    /// Remove a key-value pair from the map. Return true if the key
    /// was present in the map, otherwise false.
    #[inline(always)]
    fn remove(&mut self, key: &K) -> bool {
        self.pop(key).is_some()
    }

    /// Removes a key from the map, returning the value at the key if the key
    /// was previously in the map.
    fn pop(&mut self, key: &K) -> Option<V> {
        if self.root.is_none() {
            return None;
        }

        self.splay(key);
        let root = util::replace(&mut self.root, None).unwrap();
        if !key.equals(&root.key) {
            self.root = Some(root);
            return None;
        }

        let (value, left, right) = match root {
            ~Node {left, right, value, _} => (value, left, right)
        };

        if left.is_none() {
            self.root = right;
        } else {
            self.root = left;
            self.splay(key);
            match self.root {
                Some(ref mut node) => { node.right = right; }
                None => fail!()
            }
        }

        self.size -= 1;
        return Some(value);
    }

    /// Return a mutable reference to the value corresponding to the key
    fn find_mut<'a>(&'a mut self, key: &K) -> Option<&'a mut V> {
        if self.root.is_none() {
            return None;
        }

        self.splay(key);
        match self.root {
            None => fail!(),
            Some(ref mut r) => {
                if key.equals(&r.key) {
                    return Some(&mut r.value);
                }
                return None;
            }
        }
    }

    /// Return a reference to the value corresponding to the key
    fn find<'a>(&'a self, key: &K) -> Option<&'a V> {
        // Splay trees are self-modifying, so they can't exactly operate with
        // the immutable self given by the Map interface for this method. It can
        // be guaranteed, however, that the callers of this method are not
        // modifying the tree, they may just be rotating it. Each node is a
        // pointer on the heap, so we know that none of the pointers inside
        // these nodes (the value returned from this function) will be moving.
        //
        // With this in mind, we can unsafely use a mutable version of this tree
        // to invoke the splay operation and return a pointer to the inside of
        // one of the nodes (the pointer won't be deallocated or moved).
        unsafe {
            let self = cast::transmute_mut(self);
            match self.find_mut(key) {
                None => None,
                Some(ptr) => Some(cast::transmute_immut(ptr))
            }
        }
    }
}

impl<T> Container for SplaySet<T> {
    #[inline(always)]
    fn len(&const self) -> uint { self.map.len() }
    #[inline(always)]
    fn is_empty(&const self) -> bool { self.map.is_empty() }
}

impl<T> Mutable for SplaySet<T> {
    #[inline(always)]
    fn clear(&mut self) { self.map.clear() }
}

impl<T: TotalOrd> Set<T> for SplaySet<T> {
    /// Add a value to the set. Return true if the value was not already
    /// present in the set.
    #[inline(always)]
    fn insert(&mut self, t: T) -> bool { self.map.insert(t, ()) }

    /// Return true if the set contains a value
    #[inline(always)]
    fn contains(&self, t: &T) -> bool { self.map.contains_key(t) }

    /// Remove a value from the set. Return true if the value was
    /// present in the set.
    #[inline(always)]
    fn remove(&mut self, t: &T) -> bool { self.map.remove(t) }

    /// Return true if the set has no elements in common with `other`.
    /// This is equivalent to checking for an empty intersection.
    fn is_disjoint(&self, other: &SplaySet<T>) -> bool {
        for self.intersection(other) |_| {
            return false;
        }
        return true;
    }

    /// Return true if the set is a subset of another
    fn is_subset(&self, other: &SplaySet<T>) -> bool {
        let mut amt = 0;
        for self.intersection(other) |_| {
            amt += 1;
        }
        return amt == self.len();
    }

    /// Return true if the set is a superset of another
    fn is_superset(&self, other: &SplaySet<T>) -> bool {
        other.is_subset(self)
    }

    /// Visit the values (in-order) representing the difference
    fn difference(&self, other: &SplaySet<T>, f: &fn(&T) -> bool) {
        let mut it = self.iter();
        let mut it = it.differencei(other.iter());
        it.advance(f)
    }

    /// Visit the values (in-order) representing the union
    fn union(&self, other: &SplaySet<T>, f: &fn(&T) -> bool) {
        let mut it = self.iter();
        let mut it = it.unioni(other.iter());
        it.advance(f)
    }

    /// Visit the values (in-order) representing the intersection
    fn intersection(&self, other: &SplaySet<T>, f: &fn(&T) -> bool) {
        let mut it = self.iter();
        let mut it = it.intersecti(other.iter());
        it.advance(f)
    }

    /// Visit the values (in-order) representing the symmetric difference
    fn symmetric_difference(&self, other: &SplaySet<T>, f: &fn(&T) -> bool) {
        let mut it = self.iter();
        let mut it = it.xori(other.iter());
        it.advance(f)
    }
}

pub impl<T: TotalOrd> SplaySet<T> {
    /// Creates a new empty set
    pub fn new() -> SplaySet<T> {
        SplaySet { map: SplayMap::new() }
    }

    /// Get a lazy iterator over the values in the set.
    /// Requires that it be frozen (immutable).
    #[inline(always)]
    fn iter<'a>(&'a self) ->
        MapIterator<(&'a T, &'a ()), &'a T,
                    BSTIterator<'a, Node<T, ()>, (&'a T, &'a ())>> // ouch
    {
        self.map.iter().transform(|(a, _)| a)
    }
}

impl<K: cmp::TotalOrd, V> Node<K, V> {
    fn new(k: K, v: V) -> ~Node<K, V> {
        ~Node{ key: k, value: v, left: None, right: None }
    }

    /// Performs the top-down splay operation at a current node. The key is the
    /// value which is being splayed. The `l` and `r` arguments are storage
    /// locations for the traversal down the tree. Once a node is recursed on,
    /// one of the children is placed in either 'l' or 'r'.
    fn splay(~self, key: &K,
             l: &mut Option<~Node<K, V>>,
             r: &mut Option<~Node<K, V>>) -> ~Node<K, V>
    {
        assert!(r.is_none());
        assert!(l.is_none());

        // When finishing the top-down splay operation, we need to ensure that
        // `self` doesn't have any children, so move its remaining children into
        // the `l` and `r` arguments.
        fn fixup<K, V>(self: ~Node<K, V>, l: &mut Option<~Node<K, V>>,
                       r: &mut Option<~Node<K, V>>) -> ~Node<K, V> {
            let mut self = self;
            *l = util::replace(&mut self.left, None);
            *r = util::replace(&mut self.right, None);
            return self;
        }

        let mut self = self;
        match key.cmp(&self.key) {
            // Found it, yay!
            Equal => { return fixup(self, l, r); }

            Less => {
                match util::replace(&mut self.left, None) {
                    None => { return fixup(self, l, r); }
                    Some(left) => {
                        let mut left = left;
                        // rotate this node right if necessary
                        if key.cmp(&left.key) == Less {
                            self.left = util::replace(&mut left.right, None);
                            left.right = Some(self);
                            self = left;
                            match util::replace(&mut self.left, None) {
                                Some(l) => { left = l; }
                                None => { return fixup(self, l, r); }
                            }
                        }

                        // Bit of an odd way to get some loans, but it works
                        *r = Some(self);
                        match *r {
                            None => fail!(),
                            Some(ref mut me) => {
                                return left.splay(key, l, &mut me.left);
                            }
                        }
                    }
                }
            }

            // If you look closely, you may have seen some similar code before
            Greater => {
                match util::replace(&mut self.right, None) {
                    None => { return fixup(self, l, r); }
                    // rotate left if necessary
                    Some(right) => {
                        let mut right = right;
                        if key.cmp(&right.key) == Greater {
                            self.right = util::replace(&mut right.left, None);
                            right.left = Some(self);
                            self = right;
                            match util::replace(&mut self.right, None) {
                                Some(r) => { right = r; }
                                None => { return fixup(self, l, r); }
                            }
                        }

                        *l = Some(self);
                        match *l {
                            None => fail!(),
                            Some(ref mut me) => {
                                return right.splay(key, &mut me.right, r);
                            }
                        }
                    }
                }
            }
        }
    }

    fn each<'a>(&'a self, f: &fn(&K, &'a V) -> bool) {
        for self.left.each |left| {
            left.each(f);
        }
        f(&self.key, &self.value);
        for self.right.each |right| {
            right.each(f);
        }
    }

    fn each_mut(&mut self, f: &fn(&K, &mut V) -> bool) {
        match self.left {
            None => (),
            Some(ref mut left) => left.each_mut(f),
        }
        f(&self.key, &mut self.value);
        match self.right {
            None => (),
            Some(ref mut right) => right.each_mut(f),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    // Lots of these are shamelessly stolen from the TreeMap tests, it'd be
    // awesome if they could share them...

    #[test]
    fn insert_simple() {
        let mut t = SplayMap::new();
        assert!(t.insert(1, 2));
        assert!(!t.insert(1, 3));
        assert!(t.insert(2, 3));
    }

    #[test]
    fn remove_simple() {
        let mut t = SplayMap::new();
        assert!(t.insert(1, 2));
        assert!(t.insert(2, 3));
        assert!(t.insert(3, 4));
        assert!(t.insert(0, 5));
        assert!(t.remove(&1));
    }

    #[test]
    fn pop_simple() {
        let mut t = SplayMap::new();
        assert!(t.insert(1, 2));
        assert!(t.insert(2, 3));
        assert!(t.insert(3, 4));
        assert!(t.insert(0, 5));
        assert_eq!(t.pop(&1), Some(2));
        assert_eq!(t.pop(&1), None);
        assert_eq!(t.pop(&0), Some(5));
    }

    #[test]
    fn find_mut_simple() {
        let mut t = SplayMap::new();
        assert!(t.insert(1, 2));
        assert!(t.insert(2, 3));
        assert!(t.insert(3, 4));
        assert!(t.insert(0, 5));

        {
            let ptr = t.find_mut(&1);
            assert!(ptr.is_some());
            let ptr = ptr.unwrap();
            assert!(*ptr == 2);
            *ptr = 4;
        }

        let ptr = t.find_mut(&1);
        assert!(ptr.is_some());
        assert!(*ptr.unwrap() == 4);
    }

    #[test]
    fn each_simple() {
        let mut m = SplayMap::new();
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
    fn test_len() {
        let mut m = SplayMap::new();
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
    fn test_clear() {
        let mut m = SplayMap::new();
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
    fn insert_replace() {
        let mut m = SplayMap::new();
        assert!(m.insert(5, 2));
        assert!(m.insert(2, 9));
        assert!(!m.insert(2, 11));
        assert!(m.find(&2).unwrap() == &11);
    }

    #[test]
    fn find_empty() {
        let m = SplayMap::new::<int, int>();
        assert!(m.find(&5) == None);
    }

    #[test]
    fn find_not_found() {
        let mut m = SplayMap::new();
        assert!(m.insert(1, 2));
        assert!(m.insert(5, 3));
        assert!(m.insert(9, 3));
        assert!(m.find(&2) == None);
    }
}
