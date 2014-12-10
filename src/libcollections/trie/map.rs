// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Ordered maps and sets, implemented as simple tries.
use core::prelude::*;

pub use self::Entry::*;
use self::TrieNode::*;
use alloc::boxed::Box;
use core::default::Default;
use core::fmt;
use core::fmt::Show;
use core::mem::zeroed;
use core::mem;
use core::ops::{Slice, SliceMut};
use core::uint;
use core::iter;
use core::ptr;
use std::hash::{Writer, Hash};

use slice::{Items, MutItems};
use slice;

// FIXME(conventions): implement bounded iterators
// FIXME(conventions): implement into_iter
// FIXME(conventions): replace each_reverse by making iter DoubleEnded

// FIXME: #5244: need to manually update the InternalNode constructor
const SHIFT: uint = 4;
const SIZE: uint = 1 << SHIFT;
const MASK: uint = SIZE - 1;
// The number of chunks that the key is divided into. Also the maximum depth of the TrieMap.
const MAX_DEPTH: uint = uint::BITS / SHIFT;

/// A map implemented as a radix trie.
///
/// Keys are split into sequences of 4 bits, which are used to place elements in
/// 16-entry arrays which are nested to form a tree structure. Inserted elements are placed
/// as close to the top of the tree as possible. The most significant bits of the key are used to
/// assign the key to a node/bucket in the first layer. If there are no other elements keyed by
/// the same 4 bits in the first layer, a leaf node will be created in the first layer.
/// When keys coincide, the next 4 bits are used to assign the node to a bucket in the next layer,
/// with this process continuing until an empty spot is found or there are no more bits left in the
/// key. As a result, the maximum depth using 32-bit `uint` keys is 8. The worst collisions occur
/// for very small numbers. For example, 1 and 2 are identical in all but their least significant
/// 4 bits. If both numbers are used as keys, a chain of maximum length will be created to
/// differentiate them.
///
/// # Examples
///
/// ```
/// use std::collections::TrieMap;
///
/// let mut map = TrieMap::new();
/// map.insert(27, "Olaf");
/// map.insert(1, "Edgar");
/// map.insert(13, "Ruth");
/// map.insert(1, "Martin");
///
/// assert_eq!(map.len(), 3);
/// assert_eq!(map.get(&1), Some(&"Martin"));
///
/// if !map.contains_key(&90) {
///     println!("Nobody is keyed 90");
/// }
///
/// // Update a key
/// match map.get_mut(&1) {
///     Some(value) => *value = "Olga",
///     None => (),
/// }
///
/// map.remove(&13);
/// assert_eq!(map.len(), 2);
///
/// // Print the key value pairs, ordered by key.
/// for (key, value) in map.iter() {
///     // Prints `1: Olga` then `27: Olaf`
///     println!("{}: {}", key, value);
/// }
///
/// map.clear();
/// assert!(map.is_empty());
/// ```
#[deriving(Clone)]
pub struct TrieMap<T> {
    root: InternalNode<T>,
    length: uint
}

// An internal node holds SIZE child nodes, which may themselves contain more internal nodes.
//
// Throughout this implementation, "idx" is used to refer to a section of key that is used
// to access a node. The layer of the tree directly below the root corresponds to idx 0.
struct InternalNode<T> {
    // The number of direct children which are external (i.e. that store a value).
    count: uint,
    children: [TrieNode<T>, ..SIZE]
}

// Each child of an InternalNode may be internal, in which case nesting continues,
// external (containing a value), or empty
#[deriving(Clone)]
enum TrieNode<T> {
    Internal(Box<InternalNode<T>>),
    External(uint, T),
    Nothing
}

impl<T: PartialEq> PartialEq for TrieMap<T> {
    fn eq(&self, other: &TrieMap<T>) -> bool {
        self.len() == other.len() &&
            self.iter().zip(other.iter()).all(|(a, b)| a == b)
    }
}

impl<T: Eq> Eq for TrieMap<T> {}

impl<T: PartialOrd> PartialOrd for TrieMap<T> {
    #[inline]
    fn partial_cmp(&self, other: &TrieMap<T>) -> Option<Ordering> {
        iter::order::partial_cmp(self.iter(), other.iter())
    }
}

impl<T: Ord> Ord for TrieMap<T> {
    #[inline]
    fn cmp(&self, other: &TrieMap<T>) -> Ordering {
        iter::order::cmp(self.iter(), other.iter())
    }
}

impl<T: Show> Show for TrieMap<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        try!(write!(f, "{{"));

        for (i, (k, v)) in self.iter().enumerate() {
            if i != 0 { try!(write!(f, ", ")); }
            try!(write!(f, "{}: {}", k, *v));
        }

        write!(f, "}}")
    }
}

impl<T> Default for TrieMap<T> {
    #[inline]
    fn default() -> TrieMap<T> { TrieMap::new() }
}

impl<T> TrieMap<T> {
    /// Creates an empty `TrieMap`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::TrieMap;
    /// let mut map: TrieMap<&str> = TrieMap::new();
    /// ```
    #[inline]
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn new() -> TrieMap<T> {
        TrieMap{root: InternalNode::new(), length: 0}
    }

    /// Visits all key-value pairs in reverse order. Aborts traversal when `f` returns `false`.
    /// Returns `true` if `f` returns `true` for all elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::TrieMap;
    /// let map: TrieMap<&str> = [(1, "a"), (2, "b"), (3, "c")].iter().map(|&x| x).collect();
    ///
    /// let mut vec = Vec::new();
    /// assert_eq!(true, map.each_reverse(|&key, &value| { vec.push((key, value)); true }));
    /// assert_eq!(vec, vec![(3, "c"), (2, "b"), (1, "a")]);
    ///
    /// // Stop when we reach 2
    /// let mut vec = Vec::new();
    /// assert_eq!(false, map.each_reverse(|&key, &value| { vec.push(value); key != 2 }));
    /// assert_eq!(vec, vec!["c", "b"]);
    /// ```
    #[inline]
    pub fn each_reverse<'a>(&'a self, f: |&uint, &'a T| -> bool) -> bool {
        self.root.each_reverse(f)
    }

    /// Gets an iterator visiting all keys in ascending order by the keys.
    /// The iterator's element type is `uint`.
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn keys<'r>(&'r self) -> Keys<'r, T> {
        self.iter().map(|(k, _v)| k)
    }

    /// Gets an iterator visiting all values in ascending order by the keys.
    /// The iterator's element type is `&'r T`.
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn values<'r>(&'r self) -> Values<'r, T> {
        self.iter().map(|(_k, v)| v)
    }

    /// Gets an iterator over the key-value pairs in the map, ordered by keys.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::TrieMap;
    /// let map: TrieMap<&str> = [(3, "c"), (1, "a"), (2, "b")].iter().map(|&x| x).collect();
    ///
    /// for (key, value) in map.iter() {
    ///     println!("{}: {}", key, value);
    /// }
    /// ```
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn iter<'a>(&'a self) -> Entries<'a, T> {
        let mut iter = unsafe {Entries::new()};
        iter.stack[0] = self.root.children.iter();
        iter.length = 1;
        iter.remaining_min = self.length;
        iter.remaining_max = self.length;

        iter
    }

    /// Gets an iterator over the key-value pairs in the map, with the
    /// ability to mutate the values.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::TrieMap;
    /// let mut map: TrieMap<int> = [(1, 2), (2, 4), (3, 6)].iter().map(|&x| x).collect();
    ///
    /// for (key, value) in map.iter_mut() {
    ///     *value = -(key as int);
    /// }
    ///
    /// assert_eq!(map.get(&1), Some(&-1));
    /// assert_eq!(map.get(&2), Some(&-2));
    /// assert_eq!(map.get(&3), Some(&-3));
    /// ```
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn iter_mut<'a>(&'a mut self) -> MutEntries<'a, T> {
        let mut iter = unsafe {MutEntries::new()};
        iter.stack[0] = self.root.children.iter_mut();
        iter.length = 1;
        iter.remaining_min = self.length;
        iter.remaining_max = self.length;

        iter
    }

    /// Return the number of elements in the map.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::TrieMap;
    ///
    /// let mut a = TrieMap::new();
    /// assert_eq!(a.len(), 0);
    /// a.insert(1, "a");
    /// assert_eq!(a.len(), 1);
    /// ```
    #[inline]
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn len(&self) -> uint { self.length }

    /// Return true if the map contains no elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::TrieMap;
    ///
    /// let mut a = TrieMap::new();
    /// assert!(a.is_empty());
    /// a.insert(1, "a");
    /// assert!(!a.is_empty());
    /// ```
    #[inline]
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn is_empty(&self) -> bool { self.len() == 0 }

    /// Clears the map, removing all values.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::TrieMap;
    ///
    /// let mut a = TrieMap::new();
    /// a.insert(1, "a");
    /// a.clear();
    /// assert!(a.is_empty());
    /// ```
    #[inline]
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn clear(&mut self) {
        self.root = InternalNode::new();
        self.length = 0;
    }

    /// Deprecated: renamed to `get`.
    #[deprecated = "renamed to `get`"]
    pub fn find(&self, key: &uint) -> Option<&T> {
        self.get(key)
    }

    /// Returns a reference to the value corresponding to the key.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::TrieMap;
    ///
    /// let mut map = TrieMap::new();
    /// map.insert(1, "a");
    /// assert_eq!(map.get(&1), Some(&"a"));
    /// assert_eq!(map.get(&2), None);
    /// ```
    #[inline]
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn get(&self, key: &uint) -> Option<&T> {
        let mut node = &self.root;
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

    /// Returns true if the map contains a value for the specified key.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::TrieMap;
    ///
    /// let mut map = TrieMap::new();
    /// map.insert(1, "a");
    /// assert_eq!(map.contains_key(&1), true);
    /// assert_eq!(map.contains_key(&2), false);
    /// ```
    #[inline]
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn contains_key(&self, key: &uint) -> bool {
        self.get(key).is_some()
    }

    /// Deprecated: renamed to `get_mut`.
    #[deprecated = "renamed to `get_mut`"]
    pub fn find_mut(&mut self, key: &uint) -> Option<&mut T> {
        self.get_mut(key)
    }

    /// Returns a mutable reference to the value corresponding to the key.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::TrieMap;
    ///
    /// let mut map = TrieMap::new();
    /// map.insert(1, "a");
    /// match map.get_mut(&1) {
    ///     Some(x) => *x = "b",
    ///     None => (),
    /// }
    /// assert_eq!(map[1], "b");
    /// ```
    #[inline]
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn get_mut<'a>(&'a mut self, key: &uint) -> Option<&'a mut T> {
        find_mut(&mut self.root.children[chunk(*key, 0)], *key, 1)
    }

    /// Deprecated: Renamed to `insert`.
    #[deprecated = "Renamed to `insert`"]
    pub fn swap(&mut self, key: uint, value: T) -> Option<T> {
        self.insert(key, value)
    }

    /// Inserts a key-value pair from the map. If the key already had a value
    /// present in the map, that value is returned. Otherwise, `None` is returned.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::TrieMap;
    ///
    /// let mut map = TrieMap::new();
    /// assert_eq!(map.insert(37, "a"), None);
    /// assert_eq!(map.is_empty(), false);
    ///
    /// map.insert(37, "b");
    /// assert_eq!(map.insert(37, "c"), Some("b"));
    /// assert_eq!(map[37], "c");
    /// ```
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn insert(&mut self, key: uint, value: T) -> Option<T> {
        let (_, old_val) = insert(&mut self.root.count,
                                    &mut self.root.children[chunk(key, 0)],
                                    key, value, 1);
        if old_val.is_none() { self.length += 1 }
        old_val
    }

    /// Deprecated: Renamed to `remove`.
    #[deprecated = "Renamed to `remove`"]
    pub fn pop(&mut self, key: &uint) -> Option<T> {
        self.remove(key)
    }

    /// Removes a key from the map, returning the value at the key if the key
    /// was previously in the map.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::TrieMap;
    ///
    /// let mut map = TrieMap::new();
    /// map.insert(1, "a");
    /// assert_eq!(map.remove(&1), Some("a"));
    /// assert_eq!(map.remove(&1), None);
    /// ```
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn remove(&mut self, key: &uint) -> Option<T> {
        let ret = remove(&mut self.root.count,
                         &mut self.root.children[chunk(*key, 0)],
                         *key, 1);
        if ret.is_some() { self.length -= 1 }
        ret
    }
}

// FIXME #5846 we want to be able to choose between &x and &mut x
// (with many different `x`) below, so we need to optionally pass mut
// as a tt, but the only thing we can do with a `tt` is pass them to
// other macros, so this takes the `& <mutability> <operand>` token
// sequence and forces their evaluation as an expression. (see also
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
            // InternalNodes so the raw pointer is always valid.
            //
            // # For non-`mut`
            // We like sharing code so much that even a little unsafe won't
            // stop us.
            let this = $this;
            let mut node = unsafe {
                mem::transmute::<_, uint>(&this.root) as *mut InternalNode<T>
            };

            let key = $key;

            let mut it = unsafe {$iterator_name::new()};
            // everything else is zero'd, as we want.
            it.remaining_max = this.length;

            // this addr is necessary for the `Internal` pattern.
            addr!(loop {
                    let children = unsafe {addr!(& $($mut_)* (*node).children)};
                    // it.length is the current depth in the iterator and the
                    // current depth through the `uint` key we've traversed.
                    let child_id = chunk(key, it.length);
                    let (slice_idx, ret) = match children[child_id] {
                        Internal(ref $($mut_)* n) => {
                            node = unsafe {
                                mem::transmute::<_, uint>(&**n)
                                    as *mut InternalNode<T>
                            };
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
                    // push to the stack.
                    it.stack[it.length] = children.$slice_from(&slice_idx).$iter();
                    it.length += 1;
                    if ret { return it }
                })
        }
    }
}

impl<T> TrieMap<T> {
    // If `upper` is true then returns upper_bound else returns lower_bound.
    #[inline]
    fn bound<'a>(&'a self, key: uint, upper: bool) -> Entries<'a, T> {
        bound!(Entries, self = self,
               key = key, is_upper = upper,
               slice_from = slice_from_or_fail, iter = iter,
               mutability = )
    }

    /// Gets an iterator pointing to the first key-value pair whose key is not less than `key`.
    /// If all keys in the map are less than `key` an empty iterator is returned.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::TrieMap;
    /// let map: TrieMap<&str> = [(2, "a"), (4, "b"), (6, "c")].iter().map(|&x| x).collect();
    ///
    /// assert_eq!(map.lower_bound(4).next(), Some((4, &"b")));
    /// assert_eq!(map.lower_bound(5).next(), Some((6, &"c")));
    /// assert_eq!(map.lower_bound(10).next(), None);
    /// ```
    pub fn lower_bound<'a>(&'a self, key: uint) -> Entries<'a, T> {
        self.bound(key, false)
    }

    /// Gets an iterator pointing to the first key-value pair whose key is greater than `key`.
    /// If all keys in the map are not greater than `key` an empty iterator is returned.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::TrieMap;
    /// let map: TrieMap<&str> = [(2, "a"), (4, "b"), (6, "c")].iter().map(|&x| x).collect();
    ///
    /// assert_eq!(map.upper_bound(4).next(), Some((6, &"c")));
    /// assert_eq!(map.upper_bound(5).next(), Some((6, &"c")));
    /// assert_eq!(map.upper_bound(10).next(), None);
    /// ```
    pub fn upper_bound<'a>(&'a self, key: uint) -> Entries<'a, T> {
        self.bound(key, true)
    }
    // If `upper` is true then returns upper_bound else returns lower_bound.
    #[inline]
    fn bound_mut<'a>(&'a mut self, key: uint, upper: bool) -> MutEntries<'a, T> {
        bound!(MutEntries, self = self,
               key = key, is_upper = upper,
               slice_from = slice_from_or_fail_mut, iter = iter_mut,
               mutability = mut)
    }

    /// Gets an iterator pointing to the first key-value pair whose key is not less than `key`.
    /// If all keys in the map are less than `key` an empty iterator is returned.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::TrieMap;
    /// let mut map: TrieMap<&str> = [(2, "a"), (4, "b"), (6, "c")].iter().map(|&x| x).collect();
    ///
    /// assert_eq!(map.lower_bound_mut(4).next(), Some((4, &mut "b")));
    /// assert_eq!(map.lower_bound_mut(5).next(), Some((6, &mut "c")));
    /// assert_eq!(map.lower_bound_mut(10).next(), None);
    ///
    /// for (key, value) in map.lower_bound_mut(4) {
    ///     *value = "changed";
    /// }
    ///
    /// assert_eq!(map.get(&2), Some(&"a"));
    /// assert_eq!(map.get(&4), Some(&"changed"));
    /// assert_eq!(map.get(&6), Some(&"changed"));
    /// ```
    pub fn lower_bound_mut<'a>(&'a mut self, key: uint) -> MutEntries<'a, T> {
        self.bound_mut(key, false)
    }

    /// Gets an iterator pointing to the first key-value pair whose key is greater than `key`.
    /// If all keys in the map are not greater than `key` an empty iterator is returned.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::TrieMap;
    /// let mut map: TrieMap<&str> = [(2, "a"), (4, "b"), (6, "c")].iter().map(|&x| x).collect();
    ///
    /// assert_eq!(map.upper_bound_mut(4).next(), Some((6, &mut "c")));
    /// assert_eq!(map.upper_bound_mut(5).next(), Some((6, &mut "c")));
    /// assert_eq!(map.upper_bound_mut(10).next(), None);
    ///
    /// for (key, value) in map.upper_bound_mut(4) {
    ///     *value = "changed";
    /// }
    ///
    /// assert_eq!(map.get(&2), Some(&"a"));
    /// assert_eq!(map.get(&4), Some(&"b"));
    /// assert_eq!(map.get(&6), Some(&"changed"));
    /// ```
    pub fn upper_bound_mut<'a>(&'a mut self, key: uint) -> MutEntries<'a, T> {
        self.bound_mut(key, true)
    }
}

impl<T> FromIterator<(uint, T)> for TrieMap<T> {
    fn from_iter<Iter: Iterator<(uint, T)>>(iter: Iter) -> TrieMap<T> {
        let mut map = TrieMap::new();
        map.extend(iter);
        map
    }
}

impl<T> Extend<(uint, T)> for TrieMap<T> {
    fn extend<Iter: Iterator<(uint, T)>>(&mut self, mut iter: Iter) {
        for (k, v) in iter {
            self.insert(k, v);
        }
    }
}

impl<S: Writer, T: Hash<S>> Hash<S> for TrieMap<T> {
    fn hash(&self, state: &mut S) {
        for elt in self.iter() {
            elt.hash(state);
        }
    }
}

impl<T> Index<uint, T> for TrieMap<T> {
    #[inline]
    fn index<'a>(&'a self, i: &uint) -> &'a T {
        self.get(i).expect("key not present")
    }
}

impl<T> IndexMut<uint, T> for TrieMap<T> {
    #[inline]
    fn index_mut<'a>(&'a mut self, i: &uint) -> &'a mut T {
        self.get_mut(i).expect("key not present")
    }
}

impl<T:Clone> Clone for InternalNode<T> {
    #[inline]
    fn clone(&self) -> InternalNode<T> {
        let ch = &self.children;
        InternalNode {
            count: self.count,
             children: [ch[0].clone(), ch[1].clone(), ch[2].clone(), ch[3].clone(),
                        ch[4].clone(), ch[5].clone(), ch[6].clone(), ch[7].clone(),
                        ch[8].clone(), ch[9].clone(), ch[10].clone(), ch[11].clone(),
                        ch[12].clone(), ch[13].clone(), ch[14].clone(), ch[15].clone()]}
    }
}

impl<T> InternalNode<T> {
    #[inline]
    fn new() -> InternalNode<T> {
        // FIXME: #5244: [Nothing, ..SIZE] should be possible without implicit
        // copyability
        InternalNode{count: 0,
                 children: [Nothing, Nothing, Nothing, Nothing,
                            Nothing, Nothing, Nothing, Nothing,
                            Nothing, Nothing, Nothing, Nothing,
                            Nothing, Nothing, Nothing, Nothing]}
    }
}

impl<T> InternalNode<T> {
    fn each_reverse<'a>(&'a self, f: |&uint, &'a T| -> bool) -> bool {
        for elt in self.children.iter().rev() {
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
    let sh = uint::BITS - (SHIFT * (idx + 1));
    (n >> sh) & MASK
}

fn find_mut<'r, T>(child: &'r mut TrieNode<T>, key: uint, idx: uint) -> Option<&'r mut T> {
    match *child {
        External(stored, ref mut value) if stored == key => Some(value),
        External(..) => None,
        Internal(ref mut x) => find_mut(&mut x.children[chunk(key, idx)], key, idx + 1),
        Nothing => None
    }
}

/// Inserts a new node for the given key and value, at or below `start_node`.
///
/// The index (`idx`) is the index of the next node, such that the start node
/// was accessed via parent.children[chunk(key, idx - 1)].
///
/// The count is the external node counter for the start node's parent,
/// which will be incremented only if `start_node` is transformed into a *new* external node.
///
/// Returns a mutable reference to the inserted value and an optional previous value.
fn insert<'a, T>(count: &mut uint, start_node: &'a mut TrieNode<T>, key: uint, value: T, idx: uint)
    -> (&'a mut T, Option<T>) {
    // We branch twice to avoid having to do the `replace` when we
    // don't need to; this is much faster, especially for keys that
    // have long shared prefixes.
    match *start_node {
        Nothing => {
            *count += 1;
            *start_node = External(key, value);
            match *start_node {
                External(_, ref mut value_ref) => return (value_ref, None),
                _ => unreachable!()
            }
        }
        Internal(box ref mut x) => {
            return insert(&mut x.count, &mut x.children[chunk(key, idx)], key, value, idx + 1);
        }
        External(stored_key, ref mut stored_value) if stored_key == key => {
            // Swap in the new value and return the old.
            let old_value = mem::replace(stored_value, value);
            return (stored_value, Some(old_value));
        }
        _ => {}
    }

    // Conflict, an external node with differing keys.
    // We replace the old node by an internal one, then re-insert the two values beneath it.
    match mem::replace(start_node, Internal(box InternalNode::new())) {
        External(stored_key, stored_value) => {
            match *start_node {
                Internal(box ref mut new_node) => {
                    // Re-insert the old value.
                    insert(&mut new_node.count,
                           &mut new_node.children[chunk(stored_key, idx)],
                           stored_key, stored_value, idx + 1);

                    // Insert the new value, and return a reference to it directly.
                    insert(&mut new_node.count,
                           &mut new_node.children[chunk(key, idx)],
                           key, value, idx + 1)
                }
                // Value that was just copied disappeared.
                _ => unreachable!()
            }
        }
        // Logic error in previous match.
        _ => unreachable!(),
    }
}

fn remove<T>(count: &mut uint, child: &mut TrieNode<T>, key: uint,
             idx: uint) -> Option<T> {
    let (ret, this) = match *child {
      External(stored, _) if stored == key => {
        match mem::replace(child, Nothing) {
            External(_, value) => (Some(value), true),
            _ => unreachable!()
        }
      }
      External(..) => (None, false),
      Internal(box ref mut x) => {
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

/// A view into a single entry in a TrieMap, which may be vacant or occupied.
pub enum Entry<'a, T: 'a> {
    /// An occupied entry.
    Occupied(OccupiedEntry<'a, T>),
    /// A vacant entry.
    Vacant(VacantEntry<'a, T>)
}

/// A view into an occupied entry in a TrieMap.
pub struct OccupiedEntry<'a, T: 'a> {
    search_stack: SearchStack<'a, T>
}

/// A view into a vacant entry in a TrieMap.
pub struct VacantEntry<'a, T: 'a> {
    search_stack: SearchStack<'a, T>
}

/// A list of nodes encoding a path from the root of a TrieMap to a node.
///
/// Invariants:
/// * The last node is either `External` or `Nothing`.
/// * Pointers at indexes less than `length` can be safely dereferenced.
struct SearchStack<'a, T: 'a> {
    map: &'a mut TrieMap<T>,
    length: uint,
    key: uint,
    items: [*mut TrieNode<T>, ..MAX_DEPTH]
}

impl<'a, T> SearchStack<'a, T> {
    /// Creates a new search-stack with empty entries.
    fn new(map: &'a mut TrieMap<T>, key: uint) -> SearchStack<'a, T> {
        SearchStack {
            map: map,
            length: 0,
            key: key,
            items: [ptr::null_mut(), ..MAX_DEPTH]
        }
    }

    fn push(&mut self, node: *mut TrieNode<T>) {
        self.length += 1;
        self.items[self.length - 1] = node;
    }

    fn peek(&self) -> *mut TrieNode<T> {
        self.items[self.length - 1]
    }

    fn peek_ref(&self) -> &'a mut TrieNode<T> {
        unsafe {
            &mut *self.items[self.length - 1]
        }
    }

    fn pop_ref(&mut self) -> &'a mut TrieNode<T> {
        self.length -= 1;
        unsafe {
            &mut *self.items[self.length]
        }
    }

    fn is_empty(&self) -> bool {
        self.length == 0
    }

    fn get_ref(&self, idx: uint) -> &'a mut TrieNode<T> {
        assert!(idx < self.length);
        unsafe {
            &mut *self.items[idx]
        }
    }
}

// Implementation of SearchStack creation logic.
// Once a SearchStack has been created the Entry methods are relatively straight-forward.
impl<T> TrieMap<T> {
    /// Gets the given key's corresponding entry in the map for in-place manipulation.
    #[inline]
    pub fn entry<'a>(&'a mut self, key: uint) -> Entry<'a, T> {
        // Create an empty search stack.
        let mut search_stack = SearchStack::new(self, key);

        // Unconditionally add the corresponding node from the first layer.
        let first_node = &mut search_stack.map.root.children[chunk(key, 0)] as *mut _;
        search_stack.push(first_node);

        // While no appropriate slot is found, keep descending down the Trie,
        // adding nodes to the search stack.
        let search_successful: bool;
        loop {
            match unsafe { next_child(search_stack.peek(), key, search_stack.length) } {
                (Some(child), _) => search_stack.push(child),
                (None, success) => {
                    search_successful = success;
                    break;
                }
            }
        }

        if search_successful {
            Occupied(OccupiedEntry { search_stack: search_stack })
        } else {
            Vacant(VacantEntry { search_stack: search_stack })
        }
    }
}

/// Get a mutable pointer to the next child of a node, given a key and an idx.
///
/// The idx is the index of the next child, such that `node` was accessed via
/// parent.children[chunk(key, idx - 1)].
///
/// Returns a tuple with an optional mutable pointer to the next child, and
/// a boolean flag to indicate whether the external key node was found.
///
/// This function is safe only if `node` points to a valid `TrieNode`.
#[inline]
unsafe fn next_child<'a, T>(node: *mut TrieNode<T>, key: uint, idx: uint)
    -> (Option<*mut TrieNode<T>>, bool) {
    match *node {
        // If the node is internal, tell the caller to descend further.
        Internal(box ref mut node_internal) => {
            (Some(&mut node_internal.children[chunk(key, idx)] as *mut _), false)
        },
        // If the node is external or empty, the search is complete.
        // If the key doesn't match, node expansion will be done upon
        // insertion. If it does match, we've found our node.
        External(stored_key, _) if stored_key == key => (None, true),
        External(..) | Nothing => (None, false)
    }
}

// NB: All these methods assume a correctly constructed occupied entry (matching the given key).
impl<'a, T> OccupiedEntry<'a, T> {
    /// Gets a reference to the value in the entry.
    #[inline]
    pub fn get(&self) -> &T {
        match *self.search_stack.peek_ref() {
            External(_, ref value) => value,
            // Invalid SearchStack, non-external last node.
            _ => unreachable!()
        }
    }

    /// Gets a mutable reference to the value in the entry.
    #[inline]
    pub fn get_mut(&mut self) -> &mut T {
        match *self.search_stack.peek_ref() {
            External(_, ref mut value) => value,
            // Invalid SearchStack, non-external last node.
            _ => unreachable!()
        }
    }

    /// Converts the OccupiedEntry into a mutable reference to the value in the entry,
    /// with a lifetime bound to the map itself.
    #[inline]
    pub fn into_mut(self) -> &'a mut T {
        match *self.search_stack.peek_ref() {
            External(_, ref mut value) => value,
            // Invalid SearchStack, non-external last node.
            _ => unreachable!()
        }
    }

    /// Sets the value of the entry, and returns the entry's old value.
    #[inline]
    pub fn set(&mut self, value: T) -> T {
        match *self.search_stack.peek_ref() {
            External(_, ref mut stored_value) => {
                mem::replace(stored_value, value)
            }
            // Invalid SearchStack, non-external last node.
            _ => unreachable!()
        }
    }

    /// Takes the value out of the entry, and returns it.
    #[inline]
    pub fn take(self) -> T {
        // This function removes the external leaf-node, then unwinds the search-stack
        // deleting now-childless ancestors.
        let mut search_stack = self.search_stack;

        // Extract the value from the leaf-node of interest.
        let leaf_node = mem::replace(search_stack.pop_ref(), Nothing);

        let value = match leaf_node {
            External(_, value) => value,
            // Invalid SearchStack, non-external last node.
            _ => unreachable!()
        };

        // Iterate backwards through the search stack, deleting nodes if they are childless.
        // We compare each ancestor's parent count to 1 because each ancestor reached has just
        // had one of its children deleted.
        while !search_stack.is_empty() {
            let ancestor = search_stack.pop_ref();
            match *ancestor {
                Internal(ref mut internal) => {
                    // If stopping deletion, update the child count and break.
                    if internal.count != 1 {
                        internal.count -= 1;
                        break;
                    }
                }
                // Invalid SearchStack, non-internal ancestor node.
                _ => unreachable!()
            }
            *ancestor = Nothing;
        }

        // Decrement the length of the entire TrieMap, for the removed node.
        search_stack.map.length -= 1;

        value
    }
}

impl<'a, T> VacantEntry<'a, T> {
    /// Set the vacant entry to the given value.
    pub fn set(self, value: T) -> &'a mut T {
        let search_stack = self.search_stack;
        let old_length = search_stack.length;
        let key = search_stack.key;

        // Update the TrieMap's length for the new element.
        search_stack.map.length += 1;

        // If there's only 1 node in the search stack, insert a new node below it at idx 1.
        if old_length == 1 {
            // Note: Small hack to appease the borrow checker. Can't mutably borrow root.count
            let mut temp = search_stack.map.root.count;
            let (value_ref, _) = insert(&mut temp, search_stack.get_ref(0), key, value, 1);
            search_stack.map.root.count = temp;
            value_ref
        }
        // Otherwise, find the predecessor of the last stack node, and insert as normal.
        else {
            match *search_stack.get_ref(old_length - 2) {
                Internal(box ref mut parent) => {
                    let (value_ref, _) = insert(&mut parent.count,
                                                &mut parent.children[chunk(key, old_length - 1)],
                                                key, value, old_length);
                    value_ref
                }
                // Invalid SearchStack, non-internal ancestor node.
                _ => unreachable!()
            }
        }
    }
}

/// A forward iterator over a map.
pub struct Entries<'a, T:'a> {
    stack: [slice::Items<'a, TrieNode<T>>, .. MAX_DEPTH],
    length: uint,
    remaining_min: uint,
    remaining_max: uint
}

/// A forward iterator over the key-value pairs of a map, with the
/// values being mutable.
pub struct MutEntries<'a, T:'a> {
    stack: [slice::MutItems<'a, TrieNode<T>>, .. MAX_DEPTH],
    length: uint,
    remaining_min: uint,
    remaining_max: uint
}

/// A forward iterator over the keys of a map.
pub type Keys<'a, T> =
    iter::Map<'static, (uint, &'a T), uint, Entries<'a, T>>;

/// A forward iterator over the values of a map.
pub type Values<'a, T> =
    iter::Map<'static, (uint, &'a T), &'a T, Entries<'a, T>>;

// FIXME #5846: see `addr!` above.
macro_rules! item { ($i:item) => {$i}}

macro_rules! iterator_impl {
    ($name:ident,
     iter = $iter:ident,
     mutability = $($mut_:tt)*) => {
        impl<'a, T> $name<'a, T> {
            // Create new zero'd iterator. We have a thin gilding of safety by
            // using init rather than uninit, so that the worst that can happen
            // from failing to initialise correctly after calling these is a
            // segfault.
            #[cfg(target_word_size="32")]
            unsafe fn new() -> $name<'a, T> {
                $name {
                    remaining_min: 0,
                    remaining_max: 0,
                    length: 0,
                    // ick :( ... at least the compiler will tell us if we screwed up.
                    stack: [zeroed(), zeroed(), zeroed(), zeroed(), zeroed(),
                            zeroed(), zeroed(), zeroed()]
                }
            }

            #[cfg(target_word_size="64")]
            unsafe fn new() -> $name<'a, T> {
                $name {
                    remaining_min: 0,
                    remaining_max: 0,
                    length: 0,
                    stack: [zeroed(), zeroed(), zeroed(), zeroed(),
                            zeroed(), zeroed(), zeroed(), zeroed(),
                            zeroed(), zeroed(), zeroed(), zeroed(),
                            zeroed(), zeroed(), zeroed(), zeroed()]
                }
            }
        }

        item!(impl<'a, T> Iterator<(uint, &'a $($mut_)* T)> for $name<'a, T> {
                // you might wonder why we're not even trying to act within the
                // rules, and are just manipulating raw pointers like there's no
                // such thing as invalid pointers and memory unsafety. The
                // reason is performance, without doing this we can get the
                // (now replaced) bench_iter_large microbenchmark down to about
                // 30000 ns/iter (using .unsafe_get to index self.stack directly, 38000
                // ns/iter with [] checked indexing), but this smashes that down
                // to 13500 ns/iter.
                //
                // Fortunately, it's still safe...
                //
                // We have an invariant that every Internal node
                // corresponds to one push to self.stack, and one pop,
                // nested appropriately. self.stack has enough storage
                // to store the maximum depth of Internal nodes in the
                // trie (8 on 32-bit platforms, 16 on 64-bit).
                fn next(&mut self) -> Option<(uint, &'a $($mut_)* T)> {
                    let start_ptr = self.stack.as_mut_ptr();

                    unsafe {
                        // write_ptr is the next place to write to the stack.
                        // invariant: start_ptr <= write_ptr < end of the
                        // vector.
                        let mut write_ptr = start_ptr.offset(self.length as int);
                        while write_ptr != start_ptr {
                            // indexing back one is safe, since write_ptr >
                            // start_ptr now.
                            match (*write_ptr.offset(-1)).next() {
                                // exhausted this iterator (i.e. finished this
                                // Internal node), so pop from the stack.
                                //
                                // don't bother clearing the memory, because the
                                // next time we use it we'll've written to it
                                // first.
                                None => write_ptr = write_ptr.offset(-1),
                                Some(child) => {
                                    addr!(match *child {
                                            Internal(ref $($mut_)* node) => {
                                                // going down a level, so push
                                                // to the stack (this is the
                                                // write referenced above)
                                                *write_ptr = node.children.$iter();
                                                write_ptr = write_ptr.offset(1);
                                            }
                                            External(key, ref $($mut_)* value) => {
                                                self.remaining_max -= 1;
                                                if self.remaining_min > 0 {
                                                    self.remaining_min -= 1;
                                                }
                                                // store the new length of the
                                                // stack, based on our current
                                                // position.
                                                self.length = (write_ptr as uint
                                                               - start_ptr as uint) /
                                                    mem::size_of_val(&*write_ptr);

                                                return Some((key, value));
                                            }
                                            Nothing => {}
                                        })
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
            })
    }
}

iterator_impl! { Entries, iter = iter, mutability = }
iterator_impl! { MutEntries, iter = iter_mut, mutability = mut }

#[cfg(test)]
mod test {
    use std::prelude::*;
    use std::iter::range_step;
    use std::uint;
    use std::hash;

    use super::{TrieMap, InternalNode};
    use super::Entry::*;
    use super::TrieNode::*;

    fn check_integrity<T>(trie: &InternalNode<T>) {
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

    #[test]
    fn test_find_mut() {
        let mut m = TrieMap::new();
        assert!(m.insert(1u, 12i).is_none());
        assert!(m.insert(2u, 8i).is_none());
        assert!(m.insert(5u, 14i).is_none());
        let new = 100;
        match m.get_mut(&5) {
            None => panic!(), Some(x) => *x = new
        }
        assert_eq!(m.get(&5), Some(&new));
    }

    #[test]
    fn test_find_mut_missing() {
        let mut m = TrieMap::new();
        assert!(m.get_mut(&0).is_none());
        assert!(m.insert(1u, 12i).is_none());
        assert!(m.get_mut(&0).is_none());
        assert!(m.insert(2, 8).is_none());
        assert!(m.get_mut(&0).is_none());
    }

    #[test]
    fn test_step() {
        let mut trie = TrieMap::new();
        let n = 300u;

        for x in range_step(1u, n, 2) {
            assert!(trie.insert(x, x + 1).is_none());
            assert!(trie.contains_key(&x));
            check_integrity(&trie.root);
        }

        for x in range_step(0u, n, 2) {
            assert!(!trie.contains_key(&x));
            assert!(trie.insert(x, x + 1).is_none());
            check_integrity(&trie.root);
        }

        for x in range(0u, n) {
            assert!(trie.contains_key(&x));
            assert!(!trie.insert(x, x + 1).is_none());
            check_integrity(&trie.root);
        }

        for x in range_step(1u, n, 2) {
            assert!(trie.remove(&x).is_some());
            assert!(!trie.contains_key(&x));
            check_integrity(&trie.root);
        }

        for x in range_step(0u, n, 2) {
            assert!(trie.contains_key(&x));
            assert!(!trie.insert(x, x + 1).is_none());
            check_integrity(&trie.root);
        }
    }

    #[test]
    fn test_each_reverse() {
        let mut m = TrieMap::new();

        assert!(m.insert(3, 6).is_none());
        assert!(m.insert(0, 0).is_none());
        assert!(m.insert(4, 8).is_none());
        assert!(m.insert(2, 4).is_none());
        assert!(m.insert(1, 2).is_none());

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

        for x in range(uint::MAX - 10000, uint::MAX).rev() {
            m.insert(x, x / 2);
        }

        let mut n = uint::MAX - 1;
        m.each_reverse(|k, v| {
            if n == uint::MAX - 5000 { false } else {
                assert!(n > uint::MAX - 5000);

                assert_eq!(*k, n);
                assert_eq!(*v, n / 2);
                n -= 1;
                true
            }
        });
    }

    #[test]
    fn test_insert() {
        let mut m = TrieMap::new();
        assert_eq!(m.insert(1u, 2i), None);
        assert_eq!(m.insert(1u, 3i), Some(2));
        assert_eq!(m.insert(1u, 4i), Some(3));
    }

    #[test]
    fn test_remove() {
        let mut m = TrieMap::new();
        m.insert(1u, 2i);
        assert_eq!(m.remove(&1), Some(2));
        assert_eq!(m.remove(&1), None);
    }

    #[test]
    fn test_from_iter() {
        let xs = vec![(1u, 1i), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6)];

        let map: TrieMap<int> = xs.iter().map(|&x| x).collect();

        for &(k, v) in xs.iter() {
            assert_eq!(map.get(&k), Some(&v));
        }
    }

    #[test]
    fn test_keys() {
        let vec = vec![(1, 'a'), (2, 'b'), (3, 'c')];
        let map = vec.into_iter().collect::<TrieMap<char>>();
        let keys = map.keys().collect::<Vec<uint>>();
        assert_eq!(keys.len(), 3);
        assert!(keys.contains(&1));
        assert!(keys.contains(&2));
        assert!(keys.contains(&3));
    }

    #[test]
    fn test_values() {
        let vec = vec![(1, 'a'), (2, 'b'), (3, 'c')];
        let map = vec.into_iter().collect::<TrieMap<char>>();
        let values = map.values().map(|&v| v).collect::<Vec<char>>();
        assert_eq!(values.len(), 3);
        assert!(values.contains(&'a'));
        assert!(values.contains(&'b'));
        assert!(values.contains(&'c'));
    }

    #[test]
    fn test_iteration() {
        let empty_map : TrieMap<uint> = TrieMap::new();
        assert_eq!(empty_map.iter().next(), None);

        let first = uint::MAX - 10000;
        let last = uint::MAX;

        let mut map = TrieMap::new();
        for x in range(first, last).rev() {
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
        assert!(empty_map.iter_mut().next().is_none());

        let first = uint::MAX - 10000;
        let last = uint::MAX;

        let mut map = TrieMap::new();
        for x in range(first, last).rev() {
            map.insert(x, x / 2);
        }

        let mut i = 0;
        for (k, v) in map.iter_mut() {
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
            if i % step == 0 {
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
            let mut lb_it = m_lower.lower_bound_mut(i);
            let (k, v) = lb_it.next().unwrap();
            let lb = i + i % 2;
            assert_eq!(lb, k);
            *v -= k;
        }

        for i in range(0u, 198) {
            let mut ub_it = m_upper.upper_bound_mut(i);
            let (k, v) = ub_it.next().unwrap();
            let ub = i + 2 - i % 2;
            assert_eq!(ub, k);
            *v -= k;
        }

        assert!(m_lower.lower_bound_mut(199).next().is_none());
        assert!(m_upper.upper_bound_mut(198).next().is_none());

        assert!(m_lower.iter().all(|(_, &x)| x == 0));
        assert!(m_upper.iter().all(|(_, &x)| x == 0));
    }

    #[test]
    fn test_clone() {
        let mut a = TrieMap::new();

        a.insert(1, 'a');
        a.insert(2, 'b');
        a.insert(3, 'c');

        assert!(a.clone() == a);
    }

    #[test]
    fn test_eq() {
        let mut a = TrieMap::new();
        let mut b = TrieMap::new();

        assert!(a == b);
        assert!(a.insert(0, 5i).is_none());
        assert!(a != b);
        assert!(b.insert(0, 4i).is_none());
        assert!(a != b);
        assert!(a.insert(5, 19).is_none());
        assert!(a != b);
        assert!(!b.insert(0, 5).is_none());
        assert!(a != b);
        assert!(b.insert(5, 19).is_none());
        assert!(a == b);
    }

    #[test]
    fn test_lt() {
        let mut a = TrieMap::new();
        let mut b = TrieMap::new();

        assert!(!(a < b) && !(b < a));
        assert!(b.insert(2u, 5i).is_none());
        assert!(a < b);
        assert!(a.insert(2, 7).is_none());
        assert!(!(a < b) && b < a);
        assert!(b.insert(1, 0).is_none());
        assert!(b < a);
        assert!(a.insert(0, 6).is_none());
        assert!(a < b);
        assert!(a.insert(6, 2).is_none());
        assert!(a < b && !(b < a));
    }

    #[test]
    fn test_ord() {
        let mut a = TrieMap::new();
        let mut b = TrieMap::new();

        assert!(a <= b && a >= b);
        assert!(a.insert(1u, 1i).is_none());
        assert!(a > b && a >= b);
        assert!(b < a && b <= a);
        assert!(b.insert(2, 2).is_none());
        assert!(b > a && b >= a);
        assert!(a < b && a <= b);
    }

    #[test]
    fn test_hash() {
      let mut x = TrieMap::new();
      let mut y = TrieMap::new();

      assert!(hash::hash(&x) == hash::hash(&y));
      x.insert(1, 'a');
      x.insert(2, 'b');
      x.insert(3, 'c');

      y.insert(3, 'c');
      y.insert(2, 'b');
      y.insert(1, 'a');

      assert!(hash::hash(&x) == hash::hash(&y));
    }

    #[test]
    fn test_show() {
        let mut map = TrieMap::new();
        let empty: TrieMap<uint> = TrieMap::new();

        map.insert(1, 'a');
        map.insert(2, 'b');

        let map_str = format!("{}", map);

        assert!(map_str == "{1: a, 2: b}");
        assert_eq!(format!("{}", empty), "{}");
    }

    #[test]
    fn test_index() {
        let mut map = TrieMap::new();

        map.insert(1, 2i);
        map.insert(2, 1i);
        map.insert(3, 4i);

        assert_eq!(map[2], 1);
    }

    #[test]
    #[should_fail]
    fn test_index_nonexistent() {
        let mut map = TrieMap::new();

        map.insert(1, 2i);
        map.insert(2, 1i);
        map.insert(3, 4i);

        map[4];
    }

    // Number of items to insert into the map during entry tests.
    // The tests rely on it being even.
    const SQUARES_UPPER_LIM: uint = 128;

    /// Make a TrieMap storing i^2 for i in [0, 128)
    fn squares_map() -> TrieMap<uint> {
        let mut map = TrieMap::new();
        for i in range(0, SQUARES_UPPER_LIM) {
            map.insert(i, i * i);
        }
        map
    }

    #[test]
    fn test_entry_get() {
        let mut map = squares_map();

        for i in range(0, SQUARES_UPPER_LIM) {
            match map.entry(i) {
                Occupied(slot) => assert_eq!(slot.get(), &(i * i)),
                Vacant(_) => panic!("Key not found.")
            }
        }
        check_integrity(&map.root);
    }

    #[test]
    fn test_entry_get_mut() {
        let mut map = squares_map();

        // Change the entries to cubes.
        for i in range(0, SQUARES_UPPER_LIM) {
            match map.entry(i) {
                Occupied(mut e) => {
                    *e.get_mut() = i * i * i;
                }
                Vacant(_) => panic!("Key not found.")
            }
            assert_eq!(map.get(&i).unwrap(), &(i * i * i));
        }

        check_integrity(&map.root);
    }

    #[test]
    fn test_entry_into_mut() {
        let mut map = TrieMap::new();
        map.insert(3, 6u);

        let value_ref = match map.entry(3) {
            Occupied(e) => e.into_mut(),
            Vacant(_) => panic!("Entry not found.")
        };

        assert_eq!(*value_ref, 6u);
    }

    #[test]
    fn test_entry_take() {
        let mut map = squares_map();
        assert_eq!(map.len(), SQUARES_UPPER_LIM);

        // Remove every odd key, checking that the correct value is returned.
        for i in range_step(1, SQUARES_UPPER_LIM, 2) {
            match map.entry(i) {
                Occupied(e) => assert_eq!(e.take(), i * i),
                Vacant(_) => panic!("Key not found.")
            }
        }

        check_integrity(&map.root);

        // Check that the values for even keys remain unmodified.
        for i in range_step(0, SQUARES_UPPER_LIM, 2) {
            assert_eq!(map.get(&i).unwrap(), &(i * i));
        }

        assert_eq!(map.len(), SQUARES_UPPER_LIM / 2);
    }

    #[test]
    fn test_occupied_entry_set() {
        let mut map = squares_map();

        // Change all the entries to cubes.
        for i in range(0, SQUARES_UPPER_LIM) {
            match map.entry(i) {
                Occupied(mut e) => assert_eq!(e.set(i * i * i), i * i),
                Vacant(_) => panic!("Key not found.")
            }
            assert_eq!(map.get(&i).unwrap(), &(i * i * i));
        }
        check_integrity(&map.root);
    }

    #[test]
    fn test_vacant_entry_set() {
        let mut map = TrieMap::new();

        for i in range(0, SQUARES_UPPER_LIM) {
            match map.entry(i) {
                Vacant(e) => {
                    // Insert i^2.
                    let inserted_val = e.set(i * i);
                    assert_eq!(*inserted_val, i * i);

                    // Update it to i^3 using the returned mutable reference.
                    *inserted_val = i * i * i;
                },
                _ => panic!("Non-existent key found.")
            }
            assert_eq!(map.get(&i).unwrap(), &(i * i * i));
        }

        check_integrity(&map.root);
        assert_eq!(map.len(), SQUARES_UPPER_LIM);
    }

    #[test]
    fn test_single_key() {
        let mut map = TrieMap::new();
        map.insert(1, 2u);

        match map.entry(1) {
            Occupied(e) => { e.take(); },
            _ => ()
        }
    }
}

#[cfg(test)]
mod bench {
    use std::prelude::*;
    use std::rand::{weak_rng, Rng};
    use test::{Bencher, black_box};

    use super::{TrieMap, Occupied, Vacant};

    const MAP_SIZE: uint = 1000;

    fn random_map(size: uint) -> TrieMap<uint> {
        let mut map = TrieMap::<uint>::new();
        let mut rng = weak_rng();

        for _ in range(0, size) {
            map.insert(rng.gen(), rng.gen());
        }
        map
    }

    fn bench_iter(b: &mut Bencher, size: uint) {
        let map = random_map(size);
        b.iter(|| {
            for entry in map.iter() {
                black_box(entry);
            }
        });
    }

    #[bench]
    pub fn iter_20(b: &mut Bencher) {
        bench_iter(b, 20);
    }

    #[bench]
    pub fn iter_1000(b: &mut Bencher) {
        bench_iter(b, 1000);
    }

    #[bench]
    pub fn iter_100000(b: &mut Bencher) {
        bench_iter(b, 100000);
    }

    #[bench]
    fn bench_lower_bound(b: &mut Bencher) {
        let mut m = TrieMap::<uint>::new();
        let mut rng = weak_rng();
        for _ in range(0u, MAP_SIZE) {
            m.insert(rng.gen(), rng.gen());
        }

        b.iter(|| {
            for _ in range(0u, 10) {
                m.lower_bound(rng.gen());
            }
        });
    }

    #[bench]
    fn bench_upper_bound(b: &mut Bencher) {
        let mut m = TrieMap::<uint>::new();
        let mut rng = weak_rng();
        for _ in range(0u, MAP_SIZE) {
            m.insert(rng.gen(), rng.gen());
        }

        b.iter(|| {
            for _ in range(0u, 10) {
                m.upper_bound(rng.gen());
            }
        });
    }

    #[bench]
    fn bench_insert_large(b: &mut Bencher) {
        let mut m = TrieMap::<[uint, .. 10]>::new();
        let mut rng = weak_rng();

        b.iter(|| {
            for _ in range(0u, MAP_SIZE) {
                m.insert(rng.gen(), [1, .. 10]);
            }
        });
    }

    #[bench]
    fn bench_insert_large_entry(b: &mut Bencher) {
        let mut m = TrieMap::<[uint, .. 10]>::new();
        let mut rng = weak_rng();

        b.iter(|| {
            for _ in range(0u, MAP_SIZE) {
                match m.entry(rng.gen()) {
                    Occupied(mut e) => { e.set([1, ..10]); },
                    Vacant(e) => { e.set([1, ..10]); }
                }
            }
        });
    }

    #[bench]
    fn bench_insert_large_low_bits(b: &mut Bencher) {
        let mut m = TrieMap::<[uint, .. 10]>::new();
        let mut rng = weak_rng();

        b.iter(|| {
            for _ in range(0u, MAP_SIZE) {
                // only have the last few bits set.
                m.insert(rng.gen::<uint>() & 0xff_ff, [1, .. 10]);
            }
        });
    }

    #[bench]
    fn bench_insert_small(b: &mut Bencher) {
        let mut m = TrieMap::<()>::new();
        let mut rng = weak_rng();

        b.iter(|| {
            for _ in range(0u, MAP_SIZE) {
                m.insert(rng.gen(), ());
            }
        });
    }

    #[bench]
    fn bench_insert_small_low_bits(b: &mut Bencher) {
        let mut m = TrieMap::<()>::new();
        let mut rng = weak_rng();

        b.iter(|| {
            for _ in range(0u, MAP_SIZE) {
                // only have the last few bits set.
                m.insert(rng.gen::<uint>() & 0xff_ff, ());
            }
        });
    }

    #[bench]
    fn bench_get(b: &mut Bencher) {
        let map = random_map(MAP_SIZE);
        let keys: Vec<uint> = map.keys().collect();
        b.iter(|| {
            for key in keys.iter() {
                black_box(map.get(key));
            }
        });
    }

    #[bench]
    fn bench_get_entry(b: &mut Bencher) {
        let mut map = random_map(MAP_SIZE);
        let keys: Vec<uint> = map.keys().collect();
        b.iter(|| {
            for key in keys.iter() {
                match map.entry(*key) {
                    Occupied(e) => { black_box(e.get()); },
                    _ => ()
                }
            }
        });
    }

    #[bench]
    fn bench_remove(b: &mut Bencher) {
        b.iter(|| {
            let mut map = random_map(MAP_SIZE);
            let keys: Vec<uint> = map.keys().collect();
            for key in keys.iter() {
                black_box(map.remove(key));
            }
        });
    }

    #[bench]
    fn bench_remove_entry(b: &mut Bencher) {
        b.iter(|| {
            let mut map = random_map(MAP_SIZE);
            let keys: Vec<uint> = map.keys().collect();
            for key in keys.iter() {
                match map.entry(*key) {
                    Occupied(e) => { black_box(e.take()); },
                    _ => ()
                }
            }
        });
    }
}
