// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use core::prelude::*;

use alloc::boxed::Box;

use core::borrow::BorrowFrom;
use core::default::Default;
use core::fmt;
use core::fmt::Show;
use core::iter;
use core::mem::{replace, swap};
use core::ptr;
use std::hash::{Writer, Hash};

use vec::Vec;

// FIXME(conventions): implement bounded iterators
// FIXME(conventions): replace rev_iter(_mut) by making iter(_mut) DoubleEnded

/// This is implemented as an AA tree, which is a simplified variation of
/// a red-black tree where red (horizontal) nodes can only be added
/// as a right child. The time complexity is the same, and re-balancing
/// operations are more frequent but also cheaper.
///
/// # Example
///
/// ```
/// use std::collections::TreeMap;
///
/// let mut map = TreeMap::new();
///
/// map.insert(2i, "bar");
/// map.insert(1i, "foo");
/// map.insert(3i, "quux");
///
/// // In ascending order by keys
/// for (key, value) in map.iter() {
///     println!("{}: {}", key, value);
/// }
///
/// // Prints 1, 2, 3
/// for key in map.keys() {
///     println!("{}", key);
/// }
///
/// // Prints `foo`, `bar`, `quux`
/// for key in map.values() {
///     println!("{}", key);
/// }
///
/// map.remove(&1);
/// assert_eq!(map.len(), 2);
///
/// if !map.contains_key(&1) {
///     println!("1 is no more");
/// }
///
/// for key in range(0, 4) {
///     match map.get(&key) {
///         Some(val) => println!("{} has a value: {}", key, val),
///         None => println!("{} not in map", key),
///     }
/// }
///
/// map.clear();
/// assert!(map.is_empty());
/// ```
///
/// The easiest way to use `TreeMap` with a custom type as keys is to implement `Ord`.
/// We must also implement `PartialEq`, `Eq` and `PartialOrd`.
///
/// ```
/// use std::collections::TreeMap;
///
/// // We need `Eq` and `PartialEq`, these can be derived.
/// #[deriving(Eq, PartialEq)]
/// struct Troll<'a> {
///     name: &'a str,
///     level: uint,
/// }
///
/// // Implement `Ord` and sort trolls by level.
/// impl<'a> Ord for Troll<'a> {
///     fn cmp(&self, other: &Troll) -> Ordering {
///         // If we swap `self` and `other`, we get descending ordering.
///         self.level.cmp(&other.level)
///     }
/// }
///
/// // `PartialOrd` needs to be implemented as well.
/// impl<'a> PartialOrd for Troll<'a> {
///     fn partial_cmp(&self, other: &Troll) -> Option<Ordering> {
///         Some(self.cmp(other))
///     }
/// }
///
/// // Use a map to store trolls, sorted by level, and track a list of
/// // heroes slain.
/// let mut trolls = TreeMap::new();
///
/// trolls.insert(Troll { name: "Orgarr", level: 2 },
///               vec!["King Karl"]);
/// trolls.insert(Troll { name: "Blargarr", level: 3 },
///               vec!["Odd"]);
/// trolls.insert(Troll { name: "Kron the Smelly One", level: 4 },
///               vec!["Omar the Brave", "Peter: Slayer of Trolls"]);
/// trolls.insert(Troll { name: "Wartilda", level: 1 },
///               vec![]);
///
/// println!("You are facing {} trolls!", trolls.len());
///
/// // Print the trolls, ordered by level with smallest level first
/// for (troll, heroes) in trolls.iter() {
///     let what = if heroes.len() == 1u { "hero" }
///                else { "heroes" };
///
///     println!("level {}: '{}' has slain {} {}",
///              troll.level, troll.name, heroes.len(), what);
/// }
///
/// // Kill all trolls
/// trolls.clear();
/// assert_eq!(trolls.len(), 0);
/// ```

// Future improvements:

// range search - O(log n) retrieval of an iterator from some key

// (possibly) implement the overloads Python does for sets:
//   * intersection: &
//   * difference: -
//   * symmetric difference: ^
//   * union: |
// These would be convenient since the methods work like `each`

#[deriving(Clone)]
pub struct TreeMap<K, V> {
    root: Option<Box<TreeNode<K, V>>>,
    length: uint
}

impl<K: PartialEq + Ord, V: PartialEq> PartialEq for TreeMap<K, V> {
    fn eq(&self, other: &TreeMap<K, V>) -> bool {
        self.len() == other.len() &&
            self.iter().zip(other.iter()).all(|(a, b)| a == b)
    }
}

impl<K: Eq + Ord, V: Eq> Eq for TreeMap<K, V> {}

impl<K: Ord, V: PartialOrd> PartialOrd for TreeMap<K, V> {
    #[inline]
    fn partial_cmp(&self, other: &TreeMap<K, V>) -> Option<Ordering> {
        iter::order::partial_cmp(self.iter(), other.iter())
    }
}

impl<K: Ord, V: Ord> Ord for TreeMap<K, V> {
    #[inline]
    fn cmp(&self, other: &TreeMap<K, V>) -> Ordering {
        iter::order::cmp(self.iter(), other.iter())
    }
}

impl<K: Ord + Show, V: Show> Show for TreeMap<K, V> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        try!(write!(f, "{{"));

        for (i, (k, v)) in self.iter().enumerate() {
            if i != 0 { try!(write!(f, ", ")); }
            try!(write!(f, "{}: {}", *k, *v));
        }

        write!(f, "}}")
    }
}

impl<K: Ord, V> Default for TreeMap<K,V> {
    #[inline]
    fn default() -> TreeMap<K, V> { TreeMap::new() }
}

impl<K: Ord, Sized? Q, V> Index<Q, V> for TreeMap<K, V> where Q: BorrowFrom<K> + Ord {
    #[inline]
    fn index<'a>(&'a self, i: &Q) -> &'a V {
        self.get(i).expect("no entry found for key")
    }
}

impl<K: Ord, Sized? Q, V> IndexMut<Q, V> for TreeMap<K, V> where Q: BorrowFrom<K> + Ord {
    #[inline]
    fn index_mut<'a>(&'a mut self, i: &Q) -> &'a mut V {
        self.get_mut(i).expect("no entry found for key")
    }
}

impl<K: Ord, V> TreeMap<K, V> {
    /// Creates an empty `TreeMap`.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::TreeMap;
    /// let mut map: TreeMap<&str, int> = TreeMap::new();
    /// ```
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn new() -> TreeMap<K, V> { TreeMap{root: None, length: 0} }

    /// Gets a lazy iterator over the keys in the map, in ascending order.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::TreeMap;
    /// let mut map = TreeMap::new();
    /// map.insert("a", 1i);
    /// map.insert("c", 3i);
    /// map.insert("b", 2i);
    ///
    /// // Print "a", "b", "c" in order.
    /// for x in map.keys() {
    ///     println!("{}", x);
    /// }
    /// ```
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn keys<'a>(&'a self) -> Keys<'a, K, V> {
        self.iter().map(|(k, _v)| k)
    }

    /// Gets a lazy iterator over the values in the map, in ascending order
    /// with respect to the corresponding keys.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::TreeMap;
    /// let mut map = TreeMap::new();
    /// map.insert("a", 1i);
    /// map.insert("c", 3i);
    /// map.insert("b", 2i);
    ///
    /// // Print 1, 2, 3 ordered by keys.
    /// for x in map.values() {
    ///     println!("{}", x);
    /// }
    /// ```
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn values<'a>(&'a self) -> Values<'a, K, V> {
        self.iter().map(|(_k, v)| v)
    }

    /// Gets a lazy iterator over the key-value pairs in the map, in ascending order.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::TreeMap;
    /// let mut map = TreeMap::new();
    /// map.insert("a", 1i);
    /// map.insert("c", 3i);
    /// map.insert("b", 2i);
    ///
    /// // Print contents in ascending order
    /// for (key, value) in map.iter() {
    ///     println!("{}: {}", key, value);
    /// }
    /// ```
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn iter<'a>(&'a self) -> Entries<'a, K, V> {
        Entries {
            stack: vec!(),
            node: deref(&self.root),
            remaining_min: self.length,
            remaining_max: self.length
        }
    }

    /// Gets a lazy reverse iterator over the key-value pairs in the map, in descending order.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::TreeMap;
    /// let mut map = TreeMap::new();
    /// map.insert("a", 1i);
    /// map.insert("c", 3i);
    /// map.insert("b", 2i);
    ///
    /// // Print contents in descending order
    /// for (key, value) in map.rev_iter() {
    ///     println!("{}: {}", key, value);
    /// }
    /// ```
    pub fn rev_iter<'a>(&'a self) -> RevEntries<'a, K, V> {
        RevEntries{iter: self.iter()}
    }

    /// Gets a lazy forward iterator over the key-value pairs in the
    /// map, with the values being mutable.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::TreeMap;
    /// let mut map = TreeMap::new();
    /// map.insert("a", 1i);
    /// map.insert("c", 3i);
    /// map.insert("b", 2i);
    ///
    /// // Add 10 until we find "b"
    /// for (key, value) in map.iter_mut() {
    ///     *value += 10;
    ///     if key == &"b" { break }
    /// }
    ///
    /// assert_eq!(map.get(&"a"), Some(&11));
    /// assert_eq!(map.get(&"b"), Some(&12));
    /// assert_eq!(map.get(&"c"), Some(&3));
    /// ```
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn iter_mut<'a>(&'a mut self) -> MutEntries<'a, K, V> {
        MutEntries {
            stack: vec!(),
            node: deref_mut(&mut self.root),
            remaining_min: self.length,
            remaining_max: self.length
        }
    }

    /// Gets a lazy reverse iterator over the key-value pairs in the
    /// map, with the values being mutable.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::TreeMap;
    /// let mut map = TreeMap::new();
    /// map.insert("a", 1i);
    /// map.insert("c", 3i);
    /// map.insert("b", 2i);
    ///
    /// // Add 10 until we find "b"
    /// for (key, value) in map.rev_iter_mut() {
    ///     *value += 10;
    ///     if key == &"b" { break }
    /// }
    ///
    /// assert_eq!(map.get(&"a"), Some(&1));
    /// assert_eq!(map.get(&"b"), Some(&12));
    /// assert_eq!(map.get(&"c"), Some(&13));
    /// ```
    pub fn rev_iter_mut<'a>(&'a mut self) -> RevMutEntries<'a, K, V> {
        RevMutEntries{iter: self.iter_mut()}
    }

    /// Gets a lazy iterator that consumes the treemap.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::TreeMap;
    /// let mut map = TreeMap::new();
    /// map.insert("a", 1i);
    /// map.insert("c", 3i);
    /// map.insert("b", 2i);
    ///
    /// // Not possible with a regular `.iter()`
    /// let vec: Vec<(&str, int)> = map.into_iter().collect();
    /// assert_eq!(vec, vec![("a", 1), ("b", 2), ("c", 3)]);
    /// ```
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn into_iter(self) -> MoveEntries<K, V> {
        let TreeMap { root, length } = self;
        let stk = match root {
            None => vec!(),
            Some(box tn) => vec!(tn)
        };
        MoveEntries {
            stack: stk,
            remaining: length
        }
    }

    /// Return the number of elements in the map.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::TreeMap;
    ///
    /// let mut a = TreeMap::new();
    /// assert_eq!(a.len(), 0);
    /// a.insert(1u, "a");
    /// assert_eq!(a.len(), 1);
    /// ```
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn len(&self) -> uint { self.length }

    /// Return true if the map contains no elements.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::TreeMap;
    ///
    /// let mut a = TreeMap::new();
    /// assert!(a.is_empty());
    /// a.insert(1u, "a");
    /// assert!(!a.is_empty());
    /// ```
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    #[inline]
    pub fn is_empty(&self) -> bool { self.len() == 0 }

    /// Clears the map, removing all values.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::TreeMap;
    ///
    /// let mut a = TreeMap::new();
    /// a.insert(1u, "a");
    /// a.clear();
    /// assert!(a.is_empty());
    /// ```
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn clear(&mut self) {
        self.root = None;
        self.length = 0
    }

    /// Deprecated: Renamed to `get`.
    #[deprecated = "Renamed to `get`"]
    pub fn find(&self, key: &K) -> Option<&V> {
        self.get(key)
    }

    /// Returns a reference to the value corresponding to the key.
    ///
    /// The key may be any borrowed form of the map's key type, but the ordering
    /// on the borrowed form *must* match the ordering on the key type.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::TreeMap;
    ///
    /// let mut map = TreeMap::new();
    /// map.insert(1u, "a");
    /// assert_eq!(map.get(&1), Some(&"a"));
    /// assert_eq!(map.get(&2), None);
    /// ```
    #[inline]
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn get<Sized? Q>(&self, key: &Q) -> Option<&V>
        where Q: BorrowFrom<K> + Ord
    {
        tree_find_with(&self.root, |k2| key.cmp(BorrowFrom::borrow_from(k2)))
    }

    /// Returns true if the map contains a value for the specified key.
    ///
    /// The key may be any borrowed form of the map's key type, but the ordering
    /// on the borrowed form *must* match the ordering on the key type.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::TreeMap;
    ///
    /// let mut map = TreeMap::new();
    /// map.insert(1u, "a");
    /// assert_eq!(map.contains_key(&1), true);
    /// assert_eq!(map.contains_key(&2), false);
    /// ```
    #[inline]
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn contains_key<Sized? Q>(&self, key: &Q) -> bool
        where Q: BorrowFrom<K> + Ord
    {
        self.get(key).is_some()
    }

    /// Deprecated: Renamed to `get_mut`.
    #[deprecated = "Renamed to `get_mut`"]
    pub fn find_mut(&mut self, key: &K) -> Option<&mut V> {
        self.get_mut(key)
    }

    /// Returns a mutable reference to the value corresponding to the key.
    ///
    /// The key may be any borrowed form of the map's key type, but the ordering
    /// on the borrowed form *must* match the ordering on the key type.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::TreeMap;
    ///
    /// let mut map = TreeMap::new();
    /// map.insert(1u, "a");
    /// match map.get_mut(&1) {
    ///     Some(x) => *x = "b",
    ///     None => (),
    /// }
    /// assert_eq!(map[1], "b");
    /// ```
    #[inline]
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn get_mut<Sized? Q>(&mut self, key: &Q) -> Option<&mut V>
        where Q: BorrowFrom<K> + Ord
    {
        tree_find_with_mut(&mut self.root, |x| key.cmp(BorrowFrom::borrow_from(x)))
    }

    /// Deprecated: Renamed to `insert`.
    #[deprecated = "Renamed to `insert`"]
    pub fn swap(&mut self, key: K, value: V) -> Option<V> {
        self.insert(key, value)
    }

    /// Inserts a key-value pair from the map. If the key already had a value
    /// present in the map, that value is returned. Otherwise, `None` is returned.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::TreeMap;
    ///
    /// let mut map = TreeMap::new();
    /// assert_eq!(map.insert(37u, "a"), None);
    /// assert_eq!(map.is_empty(), false);
    ///
    /// map.insert(37, "b");
    /// assert_eq!(map.insert(37, "c"), Some("b"));
    /// assert_eq!(map[37], "c");
    /// ```
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        let ret = insert(&mut self.root, key, value);
        if ret.is_none() { self.length += 1 }
        ret
    }

    /// Deprecated: Renamed to `remove`.
    #[deprecated = "Renamed to `remove`"]
    pub fn pop(&mut self, key: &K) -> Option<V> {
        self.remove(key)
    }

    /// Removes a key from the map, returning the value at the key if the key
    /// was previously in the map.
    ///
    /// The key may be any borrowed form of the map's key type, but the ordering
    /// on the borrowed form *must* match the ordering on the key type.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::TreeMap;
    ///
    /// let mut map = TreeMap::new();
    /// map.insert(1u, "a");
    /// assert_eq!(map.remove(&1), Some("a"));
    /// assert_eq!(map.remove(&1), None);
    /// ```
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn remove<Sized? Q>(&mut self, key: &Q) -> Option<V>
        where Q: BorrowFrom<K> + Ord
    {
        let ret = remove(&mut self.root, key);
        if ret.is_some() { self.length -= 1 }
        ret
    }
}

impl<K, V> TreeMap<K, V> {
    /// Returns the value for which `f(key)` returns `Equal`. `f` is invoked
    /// with current key and guides tree navigation. That means `f` should
    /// be aware of natural ordering of the tree.
    ///
    /// # Example
    ///
    /// ```
    /// use collections::tree_map::TreeMap;
    ///
    /// fn get_headers() -> TreeMap<String, String> {
    ///     let mut result = TreeMap::new();
    ///     result.insert("Content-Type".to_string(), "application/xml".to_string());
    ///     result.insert("User-Agent".to_string(), "Curl-Rust/0.1".to_string());
    ///     result
    /// }
    ///
    /// let headers = get_headers();
    /// let ua_key = "User-Agent";
    /// let ua = headers.find_with(|k| {
    ///    ua_key.cmp(k.as_slice())
    /// });
    ///
    /// assert_eq!((*ua.unwrap()).as_slice(), "Curl-Rust/0.1");
    /// ```
    #[inline]
    #[experimental = "likely to be renamed, may be removed"]
    pub fn find_with(&self, f:|&K| -> Ordering) -> Option<&V> {
        tree_find_with(&self.root, f)
    }

    /// Returns the value for which `f(key)` returns `Equal`. `f` is invoked
    /// with current key and guides tree navigation. That means `f` should
    /// be aware of natural ordering of the tree.
    ///
    /// # Example
    ///
    /// ```
    /// let mut t = collections::tree_map::TreeMap::new();
    /// t.insert("Content-Type", "application/xml");
    /// t.insert("User-Agent", "Curl-Rust/0.1");
    ///
    /// let new_ua = "Safari/156.0";
    /// match t.find_with_mut(|&k| "User-Agent".cmp(k)) {
    ///    Some(x) => *x = new_ua,
    ///    None => panic!(),
    /// }
    ///
    /// assert_eq!(t.get(&"User-Agent"), Some(&new_ua));
    /// ```
    #[inline]
    #[experimental = "likely to be renamed, may be removed"]
    pub fn find_with_mut<'a>(&'a mut self, f:|&K| -> Ordering) -> Option<&'a mut V> {
        tree_find_with_mut(&mut self.root, f)
    }
}

// range iterators.

macro_rules! bound_setup {
    // initialiser of the iterator to manipulate
    ($iter:expr, $k:expr,
     // whether we are looking for the lower or upper bound.
     $is_lower_bound:expr) => {
        {
            let mut iter = $iter;
            loop {
                if !iter.node.is_null() {
                    let node_k = unsafe {&(*iter.node).key};
                    match $k.cmp(node_k) {
                        Less => iter.traverse_left(),
                        Greater => iter.traverse_right(),
                        Equal => {
                            if $is_lower_bound {
                                iter.traverse_complete();
                                return iter;
                            } else {
                                iter.traverse_right()
                            }
                        }
                    }
                } else {
                    iter.traverse_complete();
                    return iter;
                }
            }
        }
    }
}


impl<K: Ord, V> TreeMap<K, V> {
    /// Gets a lazy iterator that should be initialized using
    /// `traverse_left`/`traverse_right`/`traverse_complete`.
    fn iter_for_traversal<'a>(&'a self) -> Entries<'a, K, V> {
        Entries {
            stack: vec!(),
            node: deref(&self.root),
            remaining_min: 0,
            remaining_max: self.length
        }
    }

    /// Returns a lazy iterator to the first key-value pair whose key is not less than `k`
    /// If all keys in map are less than `k` an empty iterator is returned.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::TreeMap;
    ///
    /// let mut map = TreeMap::new();
    /// map.insert(2i, "a");
    /// map.insert(4, "b");
    /// map.insert(6, "c");
    /// map.insert(8, "d");
    ///
    /// assert_eq!(map.lower_bound(&4).next(), Some((&4, &"b")));
    /// assert_eq!(map.lower_bound(&5).next(), Some((&6, &"c")));
    /// assert_eq!(map.lower_bound(&10).next(), None);
    /// ```
    pub fn lower_bound<'a>(&'a self, k: &K) -> Entries<'a, K, V> {
        bound_setup!(self.iter_for_traversal(), k, true)
    }

    /// Returns a lazy iterator to the first key-value pair whose key is greater than `k`
    /// If all keys in map are less than or equal to `k` an empty iterator is returned.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::TreeMap;
    ///
    /// let mut map = TreeMap::new();
    /// map.insert(2i, "a");
    /// map.insert(4, "b");
    /// map.insert(6, "c");
    /// map.insert(8, "d");
    ///
    /// assert_eq!(map.upper_bound(&4).next(), Some((&6, &"c")));
    /// assert_eq!(map.upper_bound(&5).next(), Some((&6, &"c")));
    /// assert_eq!(map.upper_bound(&10).next(), None);
    /// ```
    pub fn upper_bound<'a>(&'a self, k: &K) -> Entries<'a, K, V> {
        bound_setup!(self.iter_for_traversal(), k, false)
    }

    /// Gets a lazy iterator that should be initialized using
    /// `traverse_left`/`traverse_right`/`traverse_complete`.
    fn iter_mut_for_traversal<'a>(&'a mut self) -> MutEntries<'a, K, V> {
        MutEntries {
            stack: vec!(),
            node: deref_mut(&mut self.root),
            remaining_min: 0,
            remaining_max: self.length
        }
    }

    /// Returns a lazy value iterator to the first key-value pair (with
    /// the value being mutable) whose key is not less than `k`.
    ///
    /// If all keys in map are less than `k` an empty iterator is
    /// returned.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::TreeMap;
    ///
    /// let mut map = TreeMap::new();
    /// map.insert(2i, "a");
    /// map.insert(4, "b");
    /// map.insert(6, "c");
    /// map.insert(8, "d");
    ///
    /// assert_eq!(map.lower_bound_mut(&4).next(), Some((&4, &mut "b")));
    /// assert_eq!(map.lower_bound_mut(&5).next(), Some((&6, &mut "c")));
    /// assert_eq!(map.lower_bound_mut(&10).next(), None);
    ///
    /// for (key, value) in map.lower_bound_mut(&4) {
    ///     *value = "changed";
    /// }
    ///
    /// assert_eq!(map.get(&2), Some(&"a"));
    /// assert_eq!(map.get(&4), Some(&"changed"));
    /// assert_eq!(map.get(&6), Some(&"changed"));
    /// assert_eq!(map.get(&8), Some(&"changed"));
    /// ```
    pub fn lower_bound_mut<'a>(&'a mut self, k: &K) -> MutEntries<'a, K, V> {
        bound_setup!(self.iter_mut_for_traversal(), k, true)
    }

    /// Returns a lazy iterator to the first key-value pair (with the
    /// value being mutable) whose key is greater than `k`.
    ///
    /// If all keys in map are less than or equal to `k` an empty iterator
    /// is returned.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::TreeMap;
    ///
    /// let mut map = TreeMap::new();
    /// map.insert(2i, "a");
    /// map.insert(4, "b");
    /// map.insert(6, "c");
    /// map.insert(8, "d");
    ///
    /// assert_eq!(map.upper_bound_mut(&4).next(), Some((&6, &mut "c")));
    /// assert_eq!(map.upper_bound_mut(&5).next(), Some((&6, &mut "c")));
    /// assert_eq!(map.upper_bound_mut(&10).next(), None);
    ///
    /// for (key, value) in map.upper_bound_mut(&4) {
    ///     *value = "changed";
    /// }
    ///
    /// assert_eq!(map.get(&2), Some(&"a"));
    /// assert_eq!(map.get(&4), Some(&"b"));
    /// assert_eq!(map.get(&6), Some(&"changed"));
    /// assert_eq!(map.get(&8), Some(&"changed"));
    /// ```
    pub fn upper_bound_mut<'a>(&'a mut self, k: &K) -> MutEntries<'a, K, V> {
        bound_setup!(self.iter_mut_for_traversal(), k, false)
    }
}

/// Lazy forward iterator over a map
pub struct Entries<'a, K:'a, V:'a> {
    stack: Vec<&'a TreeNode<K, V>>,
    // See the comment on MutEntries; this is just to allow
    // code-sharing (for this immutable-values iterator it *could* very
    // well be Option<&'a TreeNode<K,V>>).
    node: *const TreeNode<K, V>,
    remaining_min: uint,
    remaining_max: uint
}

/// Lazy backward iterator over a map
pub struct RevEntries<'a, K:'a, V:'a> {
    iter: Entries<'a, K, V>,
}

/// Lazy forward iterator over a map that allows for the mutation of
/// the values.
pub struct MutEntries<'a, K:'a, V:'a> {
    stack: Vec<&'a mut TreeNode<K, V>>,
    // Unfortunately, we require some unsafe-ness to get around the
    // fact that we would be storing a reference *into* one of the
    // nodes in the stack.
    //
    // As far as the compiler knows, this would let us invalidate the
    // reference by assigning a new value to this node's position in
    // its parent, which would cause this current one to be
    // deallocated so this reference would be invalid. (i.e. the
    // compilers complaints are 100% correct.)
    //
    // However, as far as you humans reading this code know (or are
    // about to know, if you haven't read far enough down yet), we are
    // only reading from the TreeNode.{left,right} fields. the only
    // thing that is ever mutated is the .value field (although any
    // actual mutation that happens is done externally, by the
    // iterator consumer). So, don't be so concerned, rustc, we've got
    // it under control.
    //
    // (This field can legitimately be null.)
    node: *mut TreeNode<K, V>,
    remaining_min: uint,
    remaining_max: uint
}

/// Lazy backward iterator over a map
pub struct RevMutEntries<'a, K:'a, V:'a> {
    iter: MutEntries<'a, K, V>,
}

/// TreeMap keys iterator.
pub type Keys<'a, K, V> =
    iter::Map<'static, (&'a K, &'a V), &'a K, Entries<'a, K, V>>;

/// TreeMap values iterator.
pub type Values<'a, K, V> =
    iter::Map<'static, (&'a K, &'a V), &'a V, Entries<'a, K, V>>;


// FIXME #5846 we want to be able to choose between &x and &mut x
// (with many different `x`) below, so we need to optionally pass mut
// as a tt, but the only thing we can do with a `tt` is pass them to
// other macros, so this takes the `& <mutability> <operand>` token
// sequence and forces their evaluation as an expression.
macro_rules! addr { ($e:expr) => { $e }}
// putting an optional mut into type signatures
macro_rules! item { ($i:item) => { $i }}

macro_rules! define_iterator {
    ($name:ident,
     $rev_name:ident,

     // the function to go from &m Option<Box<TreeNode>> to *m TreeNode
     deref = $deref:ident,

     // see comment on `addr!`, this is just an optional `mut`, but
     // there's no support for 0-or-1 repeats.
     addr_mut = $($addr_mut:tt)*
     ) => {
        // private methods on the forward iterator (item!() for the
        // addr_mut in the next_ return value)
        item!(impl<'a, K, V> $name<'a, K, V> {
            #[inline(always)]
            fn next_(&mut self, forward: bool) -> Option<(&'a K, &'a $($addr_mut)* V)> {
                while !self.stack.is_empty() || !self.node.is_null() {
                    if !self.node.is_null() {
                        let node = unsafe {addr!(& $($addr_mut)* *self.node)};
                        {
                            let next_node = if forward {
                                addr!(& $($addr_mut)* node.left)
                            } else {
                                addr!(& $($addr_mut)* node.right)
                            };
                            self.node = $deref(next_node);
                        }
                        self.stack.push(node);
                    } else {
                        let node = self.stack.pop().unwrap();
                        let next_node = if forward {
                            addr!(& $($addr_mut)* node.right)
                        } else {
                            addr!(& $($addr_mut)* node.left)
                        };
                        self.node = $deref(next_node);
                        self.remaining_max -= 1;
                        if self.remaining_min > 0 {
                            self.remaining_min -= 1;
                        }
                        return Some((&node.key, addr!(& $($addr_mut)* node.value)));
                    }
                }
                None
            }

            /// traverse_left, traverse_right and traverse_complete are
            /// used to initialize Entries/MutEntries
            /// pointing to element inside tree structure.
            ///
            /// They should be used in following manner:
            ///   - create iterator using TreeMap::[mut_]iter_for_traversal
            ///   - find required node using `traverse_left`/`traverse_right`
            ///     (current node is `Entries::node` field)
            ///   - complete initialization with `traverse_complete`
            ///
            /// After this, iteration will start from `self.node`.  If
            /// `self.node` is None iteration will start from last
            /// node from which we traversed left.
            #[inline]
            fn traverse_left(&mut self) {
                let node = unsafe {addr!(& $($addr_mut)* *self.node)};
                self.node = $deref(addr!(& $($addr_mut)* node.left));
                self.stack.push(node);
            }

            #[inline]
            fn traverse_right(&mut self) {
                let node = unsafe {addr!(& $($addr_mut)* *self.node)};
                self.node = $deref(addr!(& $($addr_mut)* node.right));
            }

            #[inline]
            fn traverse_complete(&mut self) {
                if !self.node.is_null() {
                    unsafe {
                        self.stack.push(addr!(& $($addr_mut)* *self.node));
                    }
                    self.node = ptr::RawPtr::null();
                }
            }
        })

        // the forward Iterator impl.
        item!(impl<'a, K, V> Iterator<(&'a K, &'a $($addr_mut)* V)> for $name<'a, K, V> {
            /// Advances the iterator to the next node (in order) and return a
            /// tuple with a reference to the key and value. If there are no
            /// more nodes, return `None`.
            fn next(&mut self) -> Option<(&'a K, &'a $($addr_mut)* V)> {
                self.next_(true)
            }

            #[inline]
            fn size_hint(&self) -> (uint, Option<uint>) {
                (self.remaining_min, Some(self.remaining_max))
            }
        })

        // the reverse Iterator impl.
        item!(impl<'a, K, V> Iterator<(&'a K, &'a $($addr_mut)* V)> for $rev_name<'a, K, V> {
            fn next(&mut self) -> Option<(&'a K, &'a $($addr_mut)* V)> {
                self.iter.next_(false)
            }

            #[inline]
            fn size_hint(&self) -> (uint, Option<uint>) {
                self.iter.size_hint()
            }
        })
    }
} // end of define_iterator

define_iterator! {
    Entries,
    RevEntries,
    deref = deref,

    // immutable, so no mut
    addr_mut =
}
define_iterator! {
    MutEntries,
    RevMutEntries,
    deref = deref_mut,

    addr_mut = mut
}

fn deref<'a, K, V>(node: &'a Option<Box<TreeNode<K, V>>>) -> *const TreeNode<K, V> {
    match *node {
        Some(ref n) => {
            let n: &TreeNode<K, V> = &**n;
            n as *const TreeNode<K, V>
        }
        None => ptr::null()
    }
}

fn deref_mut<K, V>(x: &mut Option<Box<TreeNode<K, V>>>)
             -> *mut TreeNode<K, V> {
    match *x {
        Some(ref mut n) => {
            let n: &mut TreeNode<K, V> = &mut **n;
            n as *mut TreeNode<K, V>
        }
        None => ptr::null_mut()
    }
}

/// Lazy forward iterator over a map that consumes the map while iterating
pub struct MoveEntries<K, V> {
    stack: Vec<TreeNode<K, V>>,
    remaining: uint
}

impl<K, V> Iterator<(K, V)> for MoveEntries<K,V> {
    #[inline]
    fn next(&mut self) -> Option<(K, V)> {
        while !self.stack.is_empty() {
            let TreeNode {
                key,
                value,
                left,
                right,
                level,
            } = self.stack.pop().unwrap();

            match left {
                Some(box left) => {
                    let n = TreeNode {
                        key: key,
                        value: value,
                        left: None,
                        right: right,
                        level: level
                    };
                    self.stack.push(n);
                    self.stack.push(left);
                }
                None => {
                    match right {
                        Some(box right) => self.stack.push(right),
                        None => ()
                    }
                    self.remaining -= 1;
                    return Some((key, value))
                }
            }
        }
        None
    }

    #[inline]
    fn size_hint(&self) -> (uint, Option<uint>) {
        (self.remaining, Some(self.remaining))
    }

}



// Nodes keep track of their level in the tree, starting at 1 in the
// leaves and with a red child sharing the level of the parent.
#[deriving(Clone)]
struct TreeNode<K, V> {
    key: K,
    value: V,
    left: Option<Box<TreeNode<K, V>>>,
    right: Option<Box<TreeNode<K, V>>>,
    level: uint
}

impl<K: Ord, V> TreeNode<K, V> {
    /// Creates a new tree node.
    #[inline]
    pub fn new(key: K, value: V) -> TreeNode<K, V> {
        TreeNode{key: key, value: value, left: None, right: None, level: 1}
    }
}

// Remove left horizontal link by rotating right
fn skew<K: Ord, V>(node: &mut Box<TreeNode<K, V>>) {
    if node.left.as_ref().map_or(false, |x| x.level == node.level) {
        let mut save = node.left.take().unwrap();
        swap(&mut node.left, &mut save.right); // save.right now None
        swap(node, &mut save);
        node.right = Some(save);
    }
}

// Remove dual horizontal link by rotating left and increasing level of
// the parent
fn split<K: Ord, V>(node: &mut Box<TreeNode<K, V>>) {
    if node.right.as_ref().map_or(false,
      |x| x.right.as_ref().map_or(false, |y| y.level == node.level)) {
        let mut save = node.right.take().unwrap();
        swap(&mut node.right, &mut save.left); // save.left now None
        save.level += 1;
        swap(node, &mut save);
        node.left = Some(save);
    }
}

// Next 2 functions have the same convention: comparator gets
// at input current key and returns search_key cmp cur_key
// (i.e. search_key.cmp(&cur_key))
fn tree_find_with<'r, K, V>(node: &'r Option<Box<TreeNode<K, V>>>,
                            f: |&K| -> Ordering) -> Option<&'r V> {
    let mut current: &'r Option<Box<TreeNode<K, V>>> = node;
    loop {
        match *current {
            Some(ref r) => {
                match f(&r.key) {
                    Less => current = &r.left,
                    Greater => current = &r.right,
                    Equal => return Some(&r.value)
                }
            }
            None => return None
        }
    }
}

// See comments above tree_find_with
fn tree_find_with_mut<'r, K, V>(node: &'r mut Option<Box<TreeNode<K, V>>>,
                                f: |&K| -> Ordering) -> Option<&'r mut V> {

    let mut current = node;
    loop {
        let temp = current; // hack to appease borrowck
        match *temp {
            Some(ref mut r) => {
                match f(&r.key) {
                    Less => current = &mut r.left,
                    Greater => current = &mut r.right,
                    Equal => return Some(&mut r.value)
                }
            }
            None => return None
        }
    }
}

fn insert<K: Ord, V>(node: &mut Option<Box<TreeNode<K, V>>>,
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
       *node = Some(box TreeNode::new(key, value));
        None
      }
    }
}

fn remove<K, Sized? Q, V>(node: &mut Option<Box<TreeNode<K, V>>>, key: &Q) -> Option<V>
    where K: Ord, Q: BorrowFrom<K> + Ord
{
    fn heir_swap<K: Ord, V>(node: &mut Box<TreeNode<K, V>>,
                            child: &mut Option<Box<TreeNode<K, V>>>) {
        // *could* be done without recursion, but it won't borrow check
        for x in child.iter_mut() {
            if x.right.is_some() {
                heir_swap(node, &mut x.right);
            } else {
                swap(&mut node.key, &mut x.key);
                swap(&mut node.value, &mut x.value);
            }
        }
    }

    match *node {
      None => {
        return None; // bottom of tree
      }
      Some(ref mut save) => {
        let (ret, rebalance) = match key.cmp(BorrowFrom::borrow_from(&save.key)) {
          Less => (remove(&mut save.left, key), true),
          Greater => (remove(&mut save.right, key), true),
          Equal => {
            if save.left.is_some() {
                if save.right.is_some() {
                    let mut left = save.left.take().unwrap();
                    if left.right.is_some() {
                        heir_swap(save, &mut left.right);
                    } else {
                        swap(&mut save.key, &mut left.key);
                        swap(&mut save.value, &mut left.value);
                    }
                    save.left = Some(left);
                    (remove(&mut save.left, key), true)
                } else {
                    let new = save.left.take().unwrap();
                    let box TreeNode{value, ..} = replace(save, new);
                    *save = save.left.take().unwrap();
                    (Some(value), true)
                }
            } else if save.right.is_some() {
                let new = save.right.take().unwrap();
                let box TreeNode{value, ..} = replace(save, new);
                (Some(value), true)
            } else {
                (None, false)
            }
          }
        };

        if rebalance {
            let left_level = save.left.as_ref().map_or(0, |x| x.level);
            let right_level = save.right.as_ref().map_or(0, |x| x.level);

            // re-balance, if necessary
            if left_level < save.level - 1 || right_level < save.level - 1 {
                save.level -= 1;

                if right_level > save.level {
                    let save_level = save.level;
                    for x in save.right.iter_mut() { x.level = save_level }
                }

                skew(save);

                for right in save.right.iter_mut() {
                    skew(right);
                    for x in right.right.iter_mut() { skew(x) }
                }

                split(save);
                for x in save.right.iter_mut() { split(x) }
            }

            return ret;
        }
      }
    }
    return match node.take() {
        Some(box TreeNode{value, ..}) => Some(value), None => panic!()
    };
}

impl<K: Ord, V> FromIterator<(K, V)> for TreeMap<K, V> {
    fn from_iter<T: Iterator<(K, V)>>(iter: T) -> TreeMap<K, V> {
        let mut map = TreeMap::new();
        map.extend(iter);
        map
    }
}

impl<K: Ord, V> Extend<(K, V)> for TreeMap<K, V> {
    #[inline]
    fn extend<T: Iterator<(K, V)>>(&mut self, mut iter: T) {
        for (k, v) in iter {
            self.insert(k, v);
        }
    }
}

impl<S: Writer, K: Ord + Hash<S>, V: Hash<S>> Hash<S> for TreeMap<K, V> {
    fn hash(&self, state: &mut S) {
        for elt in self.iter() {
            elt.hash(state);
        }
    }
}


#[cfg(test)]
mod test_treemap {
    use std::prelude::*;
    use std::rand::Rng;
    use std::rand;

    use super::{TreeMap, TreeNode};

    #[test]
    fn find_empty() {
        let m: TreeMap<int,int> = TreeMap::new();
        assert!(m.get(&5) == None);
    }

    #[test]
    fn find_not_found() {
        let mut m = TreeMap::new();
        assert!(m.insert(1i, 2i).is_none());
        assert!(m.insert(5i, 3i).is_none());
        assert!(m.insert(9i, 3i).is_none());
        assert_eq!(m.get(&2), None);
    }

    #[test]
    fn find_with_empty() {
        let m: TreeMap<&'static str,int> = TreeMap::new();
        assert!(m.find_with(|&k| "test".cmp(k)) == None);
    }

    #[test]
    fn find_with_not_found() {
        let mut m = TreeMap::new();
        assert!(m.insert("test1", 2i).is_none());
        assert!(m.insert("test2", 3i).is_none());
        assert!(m.insert("test3", 3i).is_none());
        assert_eq!(m.find_with(|&k| "test4".cmp(k)), None);
    }

    #[test]
    fn find_with_found() {
        let mut m = TreeMap::new();
        assert!(m.insert("test1", 2i).is_none());
        assert!(m.insert("test2", 3i).is_none());
        assert!(m.insert("test3", 4i).is_none());
        assert_eq!(m.find_with(|&k| "test2".cmp(k)), Some(&3i));
    }

    #[test]
    fn test_find_mut() {
        let mut m = TreeMap::new();
        assert!(m.insert(1i, 12i).is_none());
        assert!(m.insert(2, 8).is_none());
        assert!(m.insert(5, 14).is_none());
        let new = 100;
        match m.get_mut(&5) {
          None => panic!(), Some(x) => *x = new
        }
        assert_eq!(m.get(&5), Some(&new));
    }

    #[test]
    fn test_find_with_mut() {
        let mut m = TreeMap::new();
        assert!(m.insert("t1", 12i).is_none());
        assert!(m.insert("t2", 8).is_none());
        assert!(m.insert("t5", 14).is_none());
        let new = 100;

        match m.find_with_mut(|&k| "t5".cmp(k)) {
          None => panic!(), Some(x) => *x = new
        }
        assert_eq!(m.find_with(|&k| "t5".cmp(k)), Some(&new));
    }

    #[test]
    fn insert_replace() {
        let mut m = TreeMap::new();
        assert!(m.insert(5i, 2i).is_none());
        assert!(m.insert(2, 9).is_none());
        assert!(!m.insert(2, 11).is_none());
        assert_eq!(m.get(&2).unwrap(), &11);
    }

    #[test]
    fn test_clear() {
        let mut m = TreeMap::new();
        m.clear();
        assert!(m.insert(5i, 11i).is_none());
        assert!(m.insert(12, -3).is_none());
        assert!(m.insert(19, 2).is_none());
        m.clear();
        assert!(m.get(&5).is_none());
        assert!(m.get(&12).is_none());
        assert!(m.get(&19).is_none());
        assert!(m.is_empty());
    }

    #[test]
    fn u8_map() {
        let mut m = TreeMap::new();

        let k1 = "foo".as_bytes();
        let k2 = "bar".as_bytes();
        let v1 = "baz".as_bytes();
        let v2 = "foobar".as_bytes();

        m.insert(k1.clone(), v1.clone());
        m.insert(k2.clone(), v2.clone());

        assert_eq!(m.get(&k2), Some(&v2));
        assert_eq!(m.get(&k1), Some(&v1));
    }

    fn check_equal<K: PartialEq + Ord, V: PartialEq>(ctrl: &[(K, V)],
                                            map: &TreeMap<K, V>) {
        assert_eq!(ctrl.is_empty(), map.is_empty());
        for x in ctrl.iter() {
            let &(ref k, ref v) = x;
            assert!(map.get(k).unwrap() == v)
        }
        for (map_k, map_v) in map.iter() {
            let mut found = false;
            for x in ctrl.iter() {
                let &(ref ctrl_k, ref ctrl_v) = x;
                if *map_k == *ctrl_k {
                    assert!(*map_v == *ctrl_v);
                    found = true;
                    break;
                }
            }
            assert!(found);
        }
    }

    fn check_left<K: Ord, V>(node: &Option<Box<TreeNode<K, V>>>,
                                  parent: &Box<TreeNode<K, V>>) {
        match *node {
          Some(ref r) => {
            assert_eq!(r.key.cmp(&parent.key), Less);
            assert!(r.level == parent.level - 1); // left is black
            check_left(&r.left, r);
            check_right(&r.right, r, false);
          }
          None => assert!(parent.level == 1) // parent is leaf
        }
    }

    fn check_right<K: Ord, V>(node: &Option<Box<TreeNode<K, V>>>,
                                   parent: &Box<TreeNode<K, V>>,
                                   parent_red: bool) {
        match *node {
          Some(ref r) => {
            assert_eq!(r.key.cmp(&parent.key), Greater);
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
        let mut map: TreeMap<int,int> = TreeMap::new();
        let mut ctrl = vec![];

        check_equal(ctrl.as_slice(), &map);
        assert!(map.get(&5).is_none());

        let seed: &[_] = &[42];
        let mut rng: rand::IsaacRng = rand::SeedableRng::from_seed(seed);

        for _ in range(0u, 3) {
            for _ in range(0u, 90) {
                let k = rng.gen();
                let v = rng.gen();
                if !ctrl.iter().any(|x| x == &(k, v)) {
                    assert!(map.insert(k, v).is_none());
                    ctrl.push((k, v));
                    check_structure(&map);
                    check_equal(ctrl.as_slice(), &map);
                }
            }

            for _ in range(0u, 30) {
                let r = rng.gen_range(0, ctrl.len());
                let (key, _) = ctrl.remove(r).unwrap();
                assert!(map.remove(&key).is_some());
                check_structure(&map);
                check_equal(ctrl.as_slice(), &map);
            }
        }
    }

    #[test]
    fn test_len() {
        let mut m = TreeMap::new();
        assert!(m.insert(3i, 6i).is_none());
        assert_eq!(m.len(), 1);
        assert!(m.insert(0, 0).is_none());
        assert_eq!(m.len(), 2);
        assert!(m.insert(4, 8).is_none());
        assert_eq!(m.len(), 3);
        assert!(m.remove(&3).is_some());
        assert_eq!(m.len(), 2);
        assert!(!m.remove(&5).is_some());
        assert_eq!(m.len(), 2);
        assert!(m.insert(2, 4).is_none());
        assert_eq!(m.len(), 3);
        assert!(m.insert(1, 2).is_none());
        assert_eq!(m.len(), 4);
    }

    #[test]
    fn test_iterator() {
        let mut m = TreeMap::new();

        assert!(m.insert(3i, 6i).is_none());
        assert!(m.insert(0, 0).is_none());
        assert!(m.insert(4, 8).is_none());
        assert!(m.insert(2, 4).is_none());
        assert!(m.insert(1, 2).is_none());

        let mut n = 0;
        for (k, v) in m.iter() {
            assert_eq!(*k, n);
            assert_eq!(*v, n * 2);
            n += 1;
        }
        assert_eq!(n, 5);
    }

    #[test]
    fn test_interval_iteration() {
        let mut m = TreeMap::new();
        for i in range(1i, 100i) {
            assert!(m.insert(i * 2, i * 4).is_none());
        }

        for i in range(1i, 198i) {
            let mut lb_it = m.lower_bound(&i);
            let (&k, &v) = lb_it.next().unwrap();
            let lb = i + i % 2;
            assert_eq!(lb, k);
            assert_eq!(lb * 2, v);

            let mut ub_it = m.upper_bound(&i);
            let (&k, &v) = ub_it.next().unwrap();
            let ub = i + 2 - i % 2;
            assert_eq!(ub, k);
            assert_eq!(ub * 2, v);
        }
        let mut end_it = m.lower_bound(&199);
        assert_eq!(end_it.next(), None);
    }

    #[test]
    fn test_rev_iter() {
        let mut m = TreeMap::new();

        assert!(m.insert(3i, 6i).is_none());
        assert!(m.insert(0, 0).is_none());
        assert!(m.insert(4, 8).is_none());
        assert!(m.insert(2, 4).is_none());
        assert!(m.insert(1, 2).is_none());

        let mut n = 4;
        for (k, v) in m.rev_iter() {
            assert_eq!(*k, n);
            assert_eq!(*v, n * 2);
            n -= 1;
        }
    }

    #[test]
    fn test_mut_iter() {
        let mut m = TreeMap::new();
        for i in range(0u, 10) {
            assert!(m.insert(i, 100 * i).is_none());
        }

        for (i, (&k, v)) in m.iter_mut().enumerate() {
            *v += k * 10 + i; // 000 + 00 + 0, 100 + 10 + 1, ...
        }

        for (&k, &v) in m.iter() {
            assert_eq!(v, 111 * k);
        }
    }
    #[test]
    fn test_mut_rev_iter() {
        let mut m = TreeMap::new();
        for i in range(0u, 10) {
            assert!(m.insert(i, 100 * i).is_none());
        }

        for (i, (&k, v)) in m.rev_iter_mut().enumerate() {
            *v += k * 10 + (9 - i); // 900 + 90 + (9 - 0), 800 + 80 + (9 - 1), ...
        }

        for (&k, &v) in m.iter() {
            assert_eq!(v, 111 * k);
        }
    }

    #[test]
    fn test_mut_interval_iter() {
        let mut m_lower = TreeMap::new();
        let mut m_upper = TreeMap::new();
        for i in range(1i, 100i) {
            assert!(m_lower.insert(i * 2, i * 4).is_none());
            assert!(m_upper.insert(i * 2, i * 4).is_none());
        }

        for i in range(1i, 199) {
            let mut lb_it = m_lower.lower_bound_mut(&i);
            let (&k, v) = lb_it.next().unwrap();
            let lb = i + i % 2;
            assert_eq!(lb, k);
            *v -= k;
        }
        for i in range(0i, 198) {
            let mut ub_it = m_upper.upper_bound_mut(&i);
            let (&k, v) = ub_it.next().unwrap();
            let ub = i + 2 - i % 2;
            assert_eq!(ub, k);
            *v -= k;
        }

        assert!(m_lower.lower_bound_mut(&199).next().is_none());

        assert!(m_upper.upper_bound_mut(&198).next().is_none());

        assert!(m_lower.iter().all(|(_, &x)| x == 0));
        assert!(m_upper.iter().all(|(_, &x)| x == 0));
    }

    #[test]
    fn test_keys() {
        let vec = vec![(1i, 'a'), (2i, 'b'), (3i, 'c')];
        let map = vec.into_iter().collect::<TreeMap<int, char>>();
        let keys = map.keys().map(|&k| k).collect::<Vec<int>>();
        assert_eq!(keys.len(), 3);
        assert!(keys.contains(&1));
        assert!(keys.contains(&2));
        assert!(keys.contains(&3));
    }

    #[test]
    fn test_values() {
        let vec = vec![(1i, 'a'), (2i, 'b'), (3i, 'c')];
        let map = vec.into_iter().collect::<TreeMap<int, char>>();
        let values = map.values().map(|&v| v).collect::<Vec<char>>();
        assert_eq!(values.len(), 3);
        assert!(values.contains(&'a'));
        assert!(values.contains(&'b'));
        assert!(values.contains(&'c'));
    }

    #[test]
    fn test_eq() {
        let mut a = TreeMap::new();
        let mut b = TreeMap::new();

        assert!(a == b);
        assert!(a.insert(0i, 5i).is_none());
        assert!(a != b);
        assert!(b.insert(0, 4).is_none());
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
        let mut a = TreeMap::new();
        let mut b = TreeMap::new();

        assert!(!(a < b) && !(b < a));
        assert!(b.insert(0i, 5i).is_none());
        assert!(a < b);
        assert!(a.insert(0, 7).is_none());
        assert!(!(a < b) && b < a);
        assert!(b.insert(-2, 0).is_none());
        assert!(b < a);
        assert!(a.insert(-5, 2).is_none());
        assert!(a < b);
        assert!(a.insert(6, 2).is_none());
        assert!(a < b && !(b < a));
    }

    #[test]
    fn test_ord() {
        let mut a = TreeMap::new();
        let mut b = TreeMap::new();

        assert!(a <= b && a >= b);
        assert!(a.insert(1i, 1i).is_none());
        assert!(a > b && a >= b);
        assert!(b < a && b <= a);
        assert!(b.insert(2, 2).is_none());
        assert!(b > a && b >= a);
        assert!(a < b && a <= b);
    }

    #[test]
    fn test_show() {
        let mut map: TreeMap<int, int> = TreeMap::new();
        let empty: TreeMap<int, int> = TreeMap::new();

        map.insert(1, 2);
        map.insert(3, 4);

        let map_str = format!("{}", map);

        assert!(map_str == "{1: 2, 3: 4}".to_string());
        assert_eq!(format!("{}", empty), "{}".to_string());
    }

    #[test]
    fn test_lazy_iterator() {
        let mut m = TreeMap::new();
        let (x1, y1) = (2i, 5i);
        let (x2, y2) = (9, 12);
        let (x3, y3) = (20, -3);
        let (x4, y4) = (29, 5);
        let (x5, y5) = (103, 3);

        assert!(m.insert(x1, y1).is_none());
        assert!(m.insert(x2, y2).is_none());
        assert!(m.insert(x3, y3).is_none());
        assert!(m.insert(x4, y4).is_none());
        assert!(m.insert(x5, y5).is_none());

        let m = m;
        let mut a = m.iter();

        assert_eq!(a.next().unwrap(), (&x1, &y1));
        assert_eq!(a.next().unwrap(), (&x2, &y2));
        assert_eq!(a.next().unwrap(), (&x3, &y3));
        assert_eq!(a.next().unwrap(), (&x4, &y4));
        assert_eq!(a.next().unwrap(), (&x5, &y5));

        assert!(a.next().is_none());

        let mut b = m.iter();

        let expected = [(&x1, &y1), (&x2, &y2), (&x3, &y3), (&x4, &y4),
                        (&x5, &y5)];
        let mut i = 0;

        for x in b {
            assert_eq!(expected[i], x);
            i += 1;

            if i == 2 {
                break
            }
        }

        for x in b {
            assert_eq!(expected[i], x);
            i += 1;
        }
    }

    #[test]
    fn test_from_iter() {
        let xs = [(1i, 1i), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6)];

        let map: TreeMap<int, int> = xs.iter().map(|&x| x).collect();

        for &(k, v) in xs.iter() {
            assert_eq!(map.get(&k), Some(&v));
        }
    }

    #[test]
    fn test_index() {
        let mut map: TreeMap<int, int> = TreeMap::new();

        map.insert(1, 2);
        map.insert(2, 1);
        map.insert(3, 4);

        assert_eq!(map[2], 1);
    }

    #[test]
    #[should_fail]
    fn test_index_nonexistent() {
        let mut map: TreeMap<int, int> = TreeMap::new();

        map.insert(1, 2);
        map.insert(2, 1);
        map.insert(3, 4);

        map[4];
    }

    #[test]
    fn test_swap() {
        let mut m = TreeMap::new();
        assert_eq!(m.insert(1u, 2i), None);
        assert_eq!(m.insert(1u, 3i), Some(2));
        assert_eq!(m.insert(1u, 4i), Some(3));
    }

    #[test]
    fn test_pop() {
        let mut m = TreeMap::new();
        m.insert(1u, 2i);
        assert_eq!(m.remove(&1), Some(2));
        assert_eq!(m.remove(&1), None);
    }
}

#[cfg(test)]
mod bench {
    use std::prelude::*;
    use std::rand::{weak_rng, Rng};
    use test::{Bencher, black_box};

    use super::TreeMap;
    use bench::{insert_rand_n, insert_seq_n, find_rand_n, find_seq_n};

    #[bench]
    pub fn insert_rand_100(b: &mut Bencher) {
        let mut m : TreeMap<uint,uint> = TreeMap::new();
        insert_rand_n(100, &mut m, b,
                      |m, i| { m.insert(i, 1); },
                      |m, i| { m.remove(&i); });
    }

    #[bench]
    pub fn insert_rand_10_000(b: &mut Bencher) {
        let mut m : TreeMap<uint,uint> = TreeMap::new();
        insert_rand_n(10_000, &mut m, b,
                      |m, i| { m.insert(i, 1); },
                      |m, i| { m.remove(&i); });
    }

    // Insert seq
    #[bench]
    pub fn insert_seq_100(b: &mut Bencher) {
        let mut m : TreeMap<uint,uint> = TreeMap::new();
        insert_seq_n(100, &mut m, b,
                     |m, i| { m.insert(i, 1); },
                     |m, i| { m.remove(&i); });
    }

    #[bench]
    pub fn insert_seq_10_000(b: &mut Bencher) {
        let mut m : TreeMap<uint,uint> = TreeMap::new();
        insert_seq_n(10_000, &mut m, b,
                     |m, i| { m.insert(i, 1); },
                     |m, i| { m.remove(&i); });
    }

    // Find rand
    #[bench]
    pub fn find_rand_100(b: &mut Bencher) {
        let mut m : TreeMap<uint,uint> = TreeMap::new();
        find_rand_n(100, &mut m, b,
                    |m, i| { m.insert(i, 1); },
                    |m, i| { m.get(&i); });
    }

    #[bench]
    pub fn find_rand_10_000(b: &mut Bencher) {
        let mut m : TreeMap<uint,uint> = TreeMap::new();
        find_rand_n(10_000, &mut m, b,
                    |m, i| { m.insert(i, 1); },
                    |m, i| { m.get(&i); });
    }

    // Find seq
    #[bench]
    pub fn find_seq_100(b: &mut Bencher) {
        let mut m : TreeMap<uint,uint> = TreeMap::new();
        find_seq_n(100, &mut m, b,
                   |m, i| { m.insert(i, 1); },
                   |m, i| { m.get(&i); });
    }

    #[bench]
    pub fn find_seq_10_000(b: &mut Bencher) {
        let mut m : TreeMap<uint,uint> = TreeMap::new();
        find_seq_n(10_000, &mut m, b,
                   |m, i| { m.insert(i, 1); },
                   |m, i| { m.get(&i); });
    }

    fn bench_iter(b: &mut Bencher, size: uint) {
        let mut map = TreeMap::<uint, uint>::new();
        let mut rng = weak_rng();

        for _ in range(0, size) {
            map.insert(rng.gen(), rng.gen());
        }

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
}
