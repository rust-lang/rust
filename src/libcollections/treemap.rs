// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Maps are collections of unique keys with corresponding values, and sets are
//! just unique keys without a corresponding value. The `Map` and `Set` traits in
//! `std::container` define the basic interface.
//!
//! This crate defines the `TreeMap` and `TreeSet` types. Their keys must implement `Ord`.
//!
//! `TreeMap`s are ordered.
//!
//! ## Example
//!
//! ```{rust}
//! use std::collections::TreeSet;
//!
//! let mut tree_set = TreeSet::new();
//!
//! tree_set.insert(2i);
//! tree_set.insert(1i);
//! tree_set.insert(3i);
//!
//! for i in tree_set.iter() {
//!    println!("{}", i) // prints 1, then 2, then 3
//! }
//! ```

use core::prelude::*;

use alloc::boxed::Box;
use core::default::Default;
use core::fmt;
use core::fmt::Show;
use core::iter::Peekable;
use core::iter;
use core::mem::{replace, swap};
use core::ptr;
use std::hash::{Writer, Hash};

use {Mutable, Set, MutableSet, MutableMap, Map, MutableSeq};
use vec::Vec;

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
///     match map.find(&key) {
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

impl<K: Ord, V> Collection for TreeMap<K, V> {
    fn len(&self) -> uint { self.length }
}

impl<K: Ord, V> Mutable for TreeMap<K, V> {
    fn clear(&mut self) {
        self.root = None;
        self.length = 0
    }
}

impl<K: Ord, V> Map<K, V> for TreeMap<K, V> {
    // See comments on tree_find_with
    #[inline]
    fn find<'a>(&'a self, key: &K) -> Option<&'a V> {
        tree_find_with(&self.root, |k2| key.cmp(k2))
    }
}

impl<K: Ord, V> MutableMap<K, V> for TreeMap<K, V> {
    // See comments on tree_find_with_mut
    #[inline]
    fn find_mut<'a>(&'a mut self, key: &K) -> Option<&'a mut V> {
        tree_find_with_mut(&mut self.root, |x| key.cmp(x))
    }

    fn swap(&mut self, key: K, value: V) -> Option<V> {
        let ret = insert(&mut self.root, key, value);
        if ret.is_none() { self.length += 1 }
        ret
    }

    fn pop(&mut self, key: &K) -> Option<V> {
        let ret = remove(&mut self.root, key);
        if ret.is_some() { self.length -= 1 }
        ret
    }
}

impl<K: Ord, V> Default for TreeMap<K,V> {
    #[inline]
    fn default() -> TreeMap<K, V> { TreeMap::new() }
}

impl<K: Ord, V> Index<K, V> for TreeMap<K, V> {
    #[inline]
    fn index<'a>(&'a self, i: &K) -> &'a V {
        self.find(i).expect("no entry found for key")
    }
}

/*impl<K: Ord, V> IndexMut<K, V> for TreeMap<K, V> {
    #[inline]
    fn index_mut<'a>(&'a mut self, i: &K) -> &'a mut V {
        self.find_mut(i).expect("no entry found for key")
    }
}*/

impl<K: Ord, V> TreeMap<K, V> {
    /// Creates an empty `TreeMap`.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::TreeMap;
    /// let mut map: TreeMap<&str, int> = TreeMap::new();
    /// ```
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

    /// Deprecated: use `iter_mut`.
    #[deprecated = "use iter_mut"]
    pub fn mut_iter<'a>(&'a mut self) -> MutEntries<'a, K, V> {
        self.iter_mut()
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
    /// assert_eq!(map.find(&"a"), Some(&11));
    /// assert_eq!(map.find(&"b"), Some(&12));
    /// assert_eq!(map.find(&"c"), Some(&3));
    /// ```
    pub fn iter_mut<'a>(&'a mut self) -> MutEntries<'a, K, V> {
        MutEntries {
            stack: vec!(),
            node: deref_mut(&mut self.root),
            remaining_min: self.length,
            remaining_max: self.length
        }
    }

    /// Deprecated: use `rev_iter_mut`.
    #[deprecated = "use rev_iter_mut"]
    pub fn mut_rev_iter<'a>(&'a mut self) -> RevMutEntries<'a, K, V> {
        self.rev_iter_mut()
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
    /// assert_eq!(map.find(&"a"), Some(&1));
    /// assert_eq!(map.find(&"b"), Some(&12));
    /// assert_eq!(map.find(&"c"), Some(&13));
    /// ```
    pub fn rev_iter_mut<'a>(&'a mut self) -> RevMutEntries<'a, K, V> {
        RevMutEntries{iter: self.iter_mut()}
    }

    /// Deprecated: use `into_iter`.
    #[deprecated = "use into_iter"]
    pub fn move_iter(self) -> MoveEntries<K, V> {
        self.into_iter()
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
    pub fn into_iter(self) -> MoveEntries<K, V> {
        let TreeMap { root: root, length: length } = self;
        let stk = match root {
            None => vec!(),
            Some(box tn) => vec!(tn)
        };
        MoveEntries {
            stack: stk,
            remaining: length
        }
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
    /// use collections::treemap::TreeMap;
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
    ///    ua_key.cmp(&k.as_slice())
    /// });
    ///
    /// assert_eq!((*ua.unwrap()).as_slice(), "Curl-Rust/0.1");
    /// ```
    #[inline]
    pub fn find_with<'a>(&'a self, f:|&K| -> Ordering) -> Option<&'a V> {
        tree_find_with(&self.root, f)
    }

    /// Deprecated: use `find_with_mut`.
    #[deprecated = "use find_with_mut"]
    pub fn find_mut_with<'a>(&'a mut self, f:|&K| -> Ordering) -> Option<&'a mut V> {
        self.find_with_mut(f)
    }

    /// Returns the value for which `f(key)` returns `Equal`. `f` is invoked
    /// with current key and guides tree navigation. That means `f` should
    /// be aware of natural ordering of the tree.
    ///
    /// # Example
    ///
    /// ```
    /// let mut t = collections::treemap::TreeMap::new();
    /// t.insert("Content-Type", "application/xml");
    /// t.insert("User-Agent", "Curl-Rust/0.1");
    ///
    /// let new_ua = "Safari/156.0";
    /// match t.find_with_mut(|k| "User-Agent".cmp(k)) {
    ///    Some(x) => *x = new_ua,
    ///    None => fail!(),
    /// }
    ///
    /// assert_eq!(t.find(&"User-Agent"), Some(&new_ua));
    /// ```
    #[inline]
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

    /// Deprecated: use `lower_bound_mut`.
    #[deprecated = "use lower_bound_mut"]
    pub fn mut_lower_bound<'a>(&'a mut self, k: &K) -> MutEntries<'a, K, V> {
        self.lower_bound_mut(k)
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
    /// assert_eq!(map.find(&2), Some(&"a"));
    /// assert_eq!(map.find(&4), Some(&"changed"));
    /// assert_eq!(map.find(&6), Some(&"changed"));
    /// assert_eq!(map.find(&8), Some(&"changed"));
    /// ```
    pub fn lower_bound_mut<'a>(&'a mut self, k: &K) -> MutEntries<'a, K, V> {
        bound_setup!(self.iter_mut_for_traversal(), k, true)
    }

    /// Deprecated: use `upper_bound_mut`.
    #[deprecated = "use upper_bound_mut"]
    pub fn mut_upper_bound<'a>(&'a mut self, k: &K) -> MutEntries<'a, K, V> {
        self.upper_bound_mut(k)
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
    /// assert_eq!(map.find(&2), Some(&"a"));
    /// assert_eq!(map.find(&4), Some(&"b"));
    /// assert_eq!(map.find(&6), Some(&"changed"));
    /// assert_eq!(map.find(&8), Some(&"changed"));
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
                key: key,
                value: value,
                left: left,
                right: right,
                level: level
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

impl<'a, T> Iterator<&'a T> for SetItems<'a, T> {
    #[inline]
    fn next(&mut self) -> Option<&'a T> {
        self.iter.next().map(|(value, _)| value)
    }
}

impl<'a, T> Iterator<&'a T> for RevSetItems<'a, T> {
    #[inline]
    fn next(&mut self) -> Option<&'a T> {
        self.iter.next().map(|(value, _)| value)
    }
}

/// An implementation of the `Set` trait on top of the `TreeMap` container. The
/// only requirement is that the type of the elements contained ascribes to the
/// `Ord` trait.
///
/// ## Example
///
/// ```{rust}
/// use std::collections::TreeSet;
///
/// let mut set = TreeSet::new();
///
/// set.insert(2i);
/// set.insert(1i);
/// set.insert(3i);
///
/// for i in set.iter() {
///    println!("{}", i) // prints 1, then 2, then 3
/// }
///
/// set.remove(&3);
///
/// if !set.contains(&3) {
///     println!("set does not contain a 3 anymore");
/// }
/// ```
///
/// The easiest way to use `TreeSet` with a custom type is to implement `Ord`.
/// We must also implement `PartialEq`, `Eq` and `PartialOrd`.
///
/// ```
/// use std::collections::TreeSet;
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
/// let mut trolls = TreeSet::new();
///
/// trolls.insert(Troll { name: "Orgarr", level: 2 });
/// trolls.insert(Troll { name: "Blargarr", level: 3 });
/// trolls.insert(Troll { name: "Kron the Smelly One", level: 4 });
/// trolls.insert(Troll { name: "Wartilda", level: 1 });
///
/// println!("You are facing {} trolls!", trolls.len());
///
/// // Print the trolls, ordered by level with smallest level first
/// for x in trolls.iter() {
///     println!("level {}: {}!", x.level, x.name);
/// }
///
/// // Kill all trolls
/// trolls.clear();
/// assert_eq!(trolls.len(), 0);
/// ```
#[deriving(Clone)]
pub struct TreeSet<T> {
    map: TreeMap<T, ()>
}

impl<T: PartialEq + Ord> PartialEq for TreeSet<T> {
    #[inline]
    fn eq(&self, other: &TreeSet<T>) -> bool { self.map == other.map }
}

impl<T: Eq + Ord> Eq for TreeSet<T> {}

impl<T: Ord> PartialOrd for TreeSet<T> {
    #[inline]
    fn partial_cmp(&self, other: &TreeSet<T>) -> Option<Ordering> {
        self.map.partial_cmp(&other.map)
    }
}

impl<T: Ord> Ord for TreeSet<T> {
    #[inline]
    fn cmp(&self, other: &TreeSet<T>) -> Ordering {
        iter::order::cmp(self.iter(), other.iter())
    }
}

impl<T: Ord + Show> Show for TreeSet<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        try!(write!(f, "{{"));

        for (i, x) in self.iter().enumerate() {
            if i != 0 { try!(write!(f, ", ")); }
            try!(write!(f, "{}", *x));
        }

        write!(f, "}}")
    }
}

impl<T: Ord> Collection for TreeSet<T> {
    #[inline]
    fn len(&self) -> uint { self.map.len() }
}

impl<T: Ord> Mutable for TreeSet<T> {
    #[inline]
    fn clear(&mut self) { self.map.clear() }
}

impl<T: Ord> Set<T> for TreeSet<T> {
    #[inline]
    fn contains(&self, value: &T) -> bool {
        self.map.contains_key(value)
    }

    fn is_disjoint(&self, other: &TreeSet<T>) -> bool {
        self.intersection(other).next().is_none()
    }

    fn is_subset(&self, other: &TreeSet<T>) -> bool {
        let mut x = self.iter();
        let mut y = other.iter();
        let mut a = x.next();
        let mut b = y.next();
        while a.is_some() {
            if b.is_none() {
                return false;
            }

            let a1 = a.unwrap();
            let b1 = b.unwrap();

            match b1.cmp(a1) {
                Less => (),
                Greater => return false,
                Equal => a = x.next(),
            }

            b = y.next();
        }
        true
    }
}

impl<T: Ord> MutableSet<T> for TreeSet<T> {
    #[inline]
    fn insert(&mut self, value: T) -> bool { self.map.insert(value, ()) }

    #[inline]
    fn remove(&mut self, value: &T) -> bool { self.map.remove(value) }
}

impl<T: Ord> Default for TreeSet<T> {
    #[inline]
    fn default() -> TreeSet<T> { TreeSet::new() }
}

impl<T: Ord> TreeSet<T> {
    /// Creates an empty `TreeSet`.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::TreeSet;
    /// let mut set: TreeSet<int> = TreeSet::new();
    /// ```
    #[inline]
    pub fn new() -> TreeSet<T> { TreeSet{map: TreeMap::new()} }

    /// Gets a lazy iterator over the values in the set, in ascending order.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::TreeSet;
    /// let set: TreeSet<int> = [1i, 4, 3, 5, 2].iter().map(|&x| x).collect();
    ///
    /// // Will print in ascending order.
    /// for x in set.iter() {
    ///     println!("{}", x);
    /// }
    /// ```
    #[inline]
    pub fn iter<'a>(&'a self) -> SetItems<'a, T> {
        SetItems{iter: self.map.iter()}
    }

    /// Gets a lazy iterator over the values in the set, in descending order.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::TreeSet;
    /// let set: TreeSet<int> = [1i, 4, 3, 5, 2].iter().map(|&x| x).collect();
    ///
    /// // Will print in descending order.
    /// for x in set.rev_iter() {
    ///     println!("{}", x);
    /// }
    /// ```
    #[inline]
    pub fn rev_iter<'a>(&'a self) -> RevSetItems<'a, T> {
        RevSetItems{iter: self.map.rev_iter()}
    }

    /// Deprecated: use `into_iter`.
    #[deprecated = "use into_iter"]
    pub fn move_iter(self) -> MoveSetItems<T> {
        self.into_iter()
    }

    /// Creates a consuming iterator, that is, one that moves each value out of the
    /// set in ascending order. The set cannot be used after calling this.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::TreeSet;
    /// let set: TreeSet<int> = [1i, 4, 3, 5, 2].iter().map(|&x| x).collect();
    ///
    /// // Not possible with a regular `.iter()`
    /// let v: Vec<int> = set.into_iter().collect();
    /// assert_eq!(v, vec![1, 2, 3, 4, 5]);
    /// ```
    #[inline]
    pub fn into_iter(self) -> MoveSetItems<T> {
        self.map.into_iter().map(|(value, _)| value)
    }

    /// Gets a lazy iterator pointing to the first value not less than `v` (greater or equal).
    /// If all elements in the set are less than `v` empty iterator is returned.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::TreeSet;
    /// let set: TreeSet<int> = [2, 4, 6, 8].iter().map(|&x| x).collect();
    ///
    /// assert_eq!(set.lower_bound(&4).next(), Some(&4));
    /// assert_eq!(set.lower_bound(&5).next(), Some(&6));
    /// assert_eq!(set.lower_bound(&10).next(), None);
    /// ```
    #[inline]
    pub fn lower_bound<'a>(&'a self, v: &T) -> SetItems<'a, T> {
        SetItems{iter: self.map.lower_bound(v)}
    }

    /// Gets a lazy iterator pointing to the first value greater than `v`.
    /// If all elements in the set are less than or equal to `v` an
    /// empty iterator is returned.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::TreeSet;
    /// let set: TreeSet<int> = [2, 4, 6, 8].iter().map(|&x| x).collect();
    ///
    /// assert_eq!(set.upper_bound(&4).next(), Some(&6));
    /// assert_eq!(set.upper_bound(&5).next(), Some(&6));
    /// assert_eq!(set.upper_bound(&10).next(), None);
    /// ```
    #[inline]
    pub fn upper_bound<'a>(&'a self, v: &T) -> SetItems<'a, T> {
        SetItems{iter: self.map.upper_bound(v)}
    }

    /// Visits the values representing the difference, in ascending order.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::TreeSet;
    ///
    /// let a: TreeSet<int> = [1, 2, 3].iter().map(|&x| x).collect();
    /// let b: TreeSet<int> = [3, 4, 5].iter().map(|&x| x).collect();
    ///
    /// // Can be seen as `a - b`.
    /// for x in a.difference(&b) {
    ///     println!("{}", x); // Print 1 then 2
    /// }
    ///
    /// let diff: TreeSet<int> = a.difference(&b).map(|&x| x).collect();
    /// assert_eq!(diff, [1, 2].iter().map(|&x| x).collect());
    ///
    /// // Note that difference is not symmetric,
    /// // and `b - a` means something else:
    /// let diff: TreeSet<int> = b.difference(&a).map(|&x| x).collect();
    /// assert_eq!(diff, [4, 5].iter().map(|&x| x).collect());
    /// ```
    pub fn difference<'a>(&'a self, other: &'a TreeSet<T>) -> DifferenceItems<'a, T> {
        DifferenceItems{a: self.iter().peekable(), b: other.iter().peekable()}
    }

    /// Visits the values representing the symmetric difference, in ascending order.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::TreeSet;
    ///
    /// let a: TreeSet<int> = [1, 2, 3].iter().map(|&x| x).collect();
    /// let b: TreeSet<int> = [3, 4, 5].iter().map(|&x| x).collect();
    ///
    /// // Print 1, 2, 4, 5 in ascending order.
    /// for x in a.symmetric_difference(&b) {
    ///     println!("{}", x);
    /// }
    ///
    /// let diff1: TreeSet<int> = a.symmetric_difference(&b).map(|&x| x).collect();
    /// let diff2: TreeSet<int> = b.symmetric_difference(&a).map(|&x| x).collect();
    ///
    /// assert_eq!(diff1, diff2);
    /// assert_eq!(diff1, [1, 2, 4, 5].iter().map(|&x| x).collect());
    /// ```
    pub fn symmetric_difference<'a>(&'a self, other: &'a TreeSet<T>)
        -> SymDifferenceItems<'a, T> {
        SymDifferenceItems{a: self.iter().peekable(), b: other.iter().peekable()}
    }

    /// Visits the values representing the intersection, in ascending order.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::TreeSet;
    ///
    /// let a: TreeSet<int> = [1, 2, 3].iter().map(|&x| x).collect();
    /// let b: TreeSet<int> = [2, 3, 4].iter().map(|&x| x).collect();
    ///
    /// // Print 2, 3 in ascending order.
    /// for x in a.intersection(&b) {
    ///     println!("{}", x);
    /// }
    ///
    /// let diff: TreeSet<int> = a.intersection(&b).map(|&x| x).collect();
    /// assert_eq!(diff, [2, 3].iter().map(|&x| x).collect());
    /// ```
    pub fn intersection<'a>(&'a self, other: &'a TreeSet<T>)
        -> IntersectionItems<'a, T> {
        IntersectionItems{a: self.iter().peekable(), b: other.iter().peekable()}
    }

    /// Visits the values representing the union, in ascending order.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::TreeSet;
    ///
    /// let a: TreeSet<int> = [1, 2, 3].iter().map(|&x| x).collect();
    /// let b: TreeSet<int> = [3, 4, 5].iter().map(|&x| x).collect();
    ///
    /// // Print 1, 2, 3, 4, 5 in ascending order.
    /// for x in a.union(&b) {
    ///     println!("{}", x);
    /// }
    ///
    /// let diff: TreeSet<int> = a.union(&b).map(|&x| x).collect();
    /// assert_eq!(diff, [1, 2, 3, 4, 5].iter().map(|&x| x).collect());
    /// ```
    pub fn union<'a>(&'a self, other: &'a TreeSet<T>) -> UnionItems<'a, T> {
        UnionItems{a: self.iter().peekable(), b: other.iter().peekable()}
    }
}

/// A lazy forward iterator over a set.
pub struct SetItems<'a, T:'a> {
    iter: Entries<'a, T, ()>
}

/// A lazy backward iterator over a set.
pub struct RevSetItems<'a, T:'a> {
    iter: RevEntries<'a, T, ()>
}

/// A lazy forward iterator over a set that consumes the set while iterating.
pub type MoveSetItems<T> = iter::Map<'static, (T, ()), T, MoveEntries<T, ()>>;

/// A lazy iterator producing elements in the set difference (in-order).
pub struct DifferenceItems<'a, T:'a> {
    a: Peekable<&'a T, SetItems<'a, T>>,
    b: Peekable<&'a T, SetItems<'a, T>>,
}

/// A lazy iterator producing elements in the set symmetric difference (in-order).
pub struct SymDifferenceItems<'a, T:'a> {
    a: Peekable<&'a T, SetItems<'a, T>>,
    b: Peekable<&'a T, SetItems<'a, T>>,
}

/// A lazy iterator producing elements in the set intersection (in-order).
pub struct IntersectionItems<'a, T:'a> {
    a: Peekable<&'a T, SetItems<'a, T>>,
    b: Peekable<&'a T, SetItems<'a, T>>,
}

/// A lazy iterator producing elements in the set union (in-order).
pub struct UnionItems<'a, T:'a> {
    a: Peekable<&'a T, SetItems<'a, T>>,
    b: Peekable<&'a T, SetItems<'a, T>>,
}

/// Compare `x` and `y`, but return `short` if x is None and `long` if y is None
fn cmp_opt<T: Ord>(x: Option<&T>, y: Option<&T>,
                        short: Ordering, long: Ordering) -> Ordering {
    match (x, y) {
        (None    , _       ) => short,
        (_       , None    ) => long,
        (Some(x1), Some(y1)) => x1.cmp(y1),
    }
}

impl<'a, T: Ord> Iterator<&'a T> for DifferenceItems<'a, T> {
    fn next(&mut self) -> Option<&'a T> {
        loop {
            match cmp_opt(self.a.peek(), self.b.peek(), Less, Less) {
                Less    => return self.a.next(),
                Equal   => { self.a.next(); self.b.next(); }
                Greater => { self.b.next(); }
            }
        }
    }
}

impl<'a, T: Ord> Iterator<&'a T> for SymDifferenceItems<'a, T> {
    fn next(&mut self) -> Option<&'a T> {
        loop {
            match cmp_opt(self.a.peek(), self.b.peek(), Greater, Less) {
                Less    => return self.a.next(),
                Equal   => { self.a.next(); self.b.next(); }
                Greater => return self.b.next(),
            }
        }
    }
}

impl<'a, T: Ord> Iterator<&'a T> for IntersectionItems<'a, T> {
    fn next(&mut self) -> Option<&'a T> {
        loop {
            let o_cmp = match (self.a.peek(), self.b.peek()) {
                (None    , _       ) => None,
                (_       , None    ) => None,
                (Some(a1), Some(b1)) => Some(a1.cmp(b1)),
            };
            match o_cmp {
                None          => return None,
                Some(Less)    => { self.a.next(); }
                Some(Equal)   => { self.b.next(); return self.a.next() }
                Some(Greater) => { self.b.next(); }
            }
        }
    }
}

impl<'a, T: Ord> Iterator<&'a T> for UnionItems<'a, T> {
    fn next(&mut self) -> Option<&'a T> {
        loop {
            match cmp_opt(self.a.peek(), self.b.peek(), Greater, Less) {
                Less    => return self.a.next(),
                Equal   => { self.b.next(); return self.a.next() }
                Greater => return self.b.next(),
            }
        }
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

fn remove<K: Ord, V>(node: &mut Option<Box<TreeNode<K, V>>>,
                          key: &K) -> Option<V> {
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
        let (ret, rebalance) = match key.cmp(&save.key) {
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
        Some(box TreeNode{value, ..}) => Some(value), None => fail!()
    };
}

impl<K: Ord, V> FromIterator<(K, V)> for TreeMap<K, V> {
    fn from_iter<T: Iterator<(K, V)>>(iter: T) -> TreeMap<K, V> {
        let mut map = TreeMap::new();
        map.extend(iter);
        map
    }
}

impl<K: Ord, V> Extendable<(K, V)> for TreeMap<K, V> {
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

impl<T: Ord> FromIterator<T> for TreeSet<T> {
    fn from_iter<Iter: Iterator<T>>(iter: Iter) -> TreeSet<T> {
        let mut set = TreeSet::new();
        set.extend(iter);
        set
    }
}

impl<T: Ord> Extendable<T> for TreeSet<T> {
    #[inline]
    fn extend<Iter: Iterator<T>>(&mut self, mut iter: Iter) {
        for elem in iter {
            self.insert(elem);
        }
    }
}

impl<S: Writer, T: Ord + Hash<S>> Hash<S> for TreeSet<T> {
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

    use {Map, MutableMap, Mutable, MutableSeq};
    use super::{TreeMap, TreeNode};

    #[test]
    fn find_empty() {
        let m: TreeMap<int,int> = TreeMap::new();
        assert!(m.find(&5) == None);
    }

    #[test]
    fn find_not_found() {
        let mut m = TreeMap::new();
        assert!(m.insert(1i, 2i));
        assert!(m.insert(5i, 3i));
        assert!(m.insert(9i, 3i));
        assert_eq!(m.find(&2), None);
    }

    #[test]
    fn find_with_empty() {
        let m: TreeMap<&'static str,int> = TreeMap::new();
        assert!(m.find_with(|k| "test".cmp(k)) == None);
    }

    #[test]
    fn find_with_not_found() {
        let mut m = TreeMap::new();
        assert!(m.insert("test1", 2i));
        assert!(m.insert("test2", 3i));
        assert!(m.insert("test3", 3i));
        assert_eq!(m.find_with(|k| "test4".cmp(k)), None);
    }

    #[test]
    fn find_with_found() {
        let mut m = TreeMap::new();
        assert!(m.insert("test1", 2i));
        assert!(m.insert("test2", 3i));
        assert!(m.insert("test3", 4i));
        assert_eq!(m.find_with(|k| "test2".cmp(k)), Some(&3i));
    }

    #[test]
    fn test_find_mut() {
        let mut m = TreeMap::new();
        assert!(m.insert(1i, 12i));
        assert!(m.insert(2, 8));
        assert!(m.insert(5, 14));
        let new = 100;
        match m.find_mut(&5) {
          None => fail!(), Some(x) => *x = new
        }
        assert_eq!(m.find(&5), Some(&new));
    }

    #[test]
    fn test_find_with_mut() {
        let mut m = TreeMap::new();
        assert!(m.insert("t1", 12i));
        assert!(m.insert("t2", 8));
        assert!(m.insert("t5", 14));
        let new = 100;
        match m.find_with_mut(|k| "t5".cmp(k)) {
          None => fail!(), Some(x) => *x = new
        }
        assert_eq!(m.find_with(|k| "t5".cmp(k)), Some(&new));
    }

    #[test]
    fn insert_replace() {
        let mut m = TreeMap::new();
        assert!(m.insert(5i, 2i));
        assert!(m.insert(2, 9));
        assert!(!m.insert(2, 11));
        assert_eq!(m.find(&2).unwrap(), &11);
    }

    #[test]
    fn test_clear() {
        let mut m = TreeMap::new();
        m.clear();
        assert!(m.insert(5i, 11i));
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

        let k1 = "foo".as_bytes();
        let k2 = "bar".as_bytes();
        let v1 = "baz".as_bytes();
        let v2 = "foobar".as_bytes();

        m.insert(k1.clone(), v1.clone());
        m.insert(k2.clone(), v2.clone());

        assert_eq!(m.find(&k2), Some(&v2));
        assert_eq!(m.find(&k1), Some(&v1));
    }

    fn check_equal<K: PartialEq + Ord, V: PartialEq>(ctrl: &[(K, V)],
                                            map: &TreeMap<K, V>) {
        assert_eq!(ctrl.is_empty(), map.is_empty());
        for x in ctrl.iter() {
            let &(ref k, ref v) = x;
            assert!(map.find(k).unwrap() == v)
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
        assert!(map.find(&5).is_none());

        let seed: &[_] = &[42];
        let mut rng: rand::IsaacRng = rand::SeedableRng::from_seed(seed);

        for _ in range(0u, 3) {
            for _ in range(0u, 90) {
                let k = rng.gen();
                let v = rng.gen();
                if !ctrl.iter().any(|x| x == &(k, v)) {
                    assert!(map.insert(k, v));
                    ctrl.push((k, v));
                    check_structure(&map);
                    check_equal(ctrl.as_slice(), &map);
                }
            }

            for _ in range(0u, 30) {
                let r = rng.gen_range(0, ctrl.len());
                let (key, _) = ctrl.remove(r).unwrap();
                assert!(map.remove(&key));
                check_structure(&map);
                check_equal(ctrl.as_slice(), &map);
            }
        }
    }

    #[test]
    fn test_len() {
        let mut m = TreeMap::new();
        assert!(m.insert(3i, 6i));
        assert_eq!(m.len(), 1);
        assert!(m.insert(0, 0));
        assert_eq!(m.len(), 2);
        assert!(m.insert(4, 8));
        assert_eq!(m.len(), 3);
        assert!(m.remove(&3));
        assert_eq!(m.len(), 2);
        assert!(!m.remove(&5));
        assert_eq!(m.len(), 2);
        assert!(m.insert(2, 4));
        assert_eq!(m.len(), 3);
        assert!(m.insert(1, 2));
        assert_eq!(m.len(), 4);
    }

    #[test]
    fn test_iterator() {
        let mut m = TreeMap::new();

        assert!(m.insert(3i, 6i));
        assert!(m.insert(0, 0));
        assert!(m.insert(4, 8));
        assert!(m.insert(2, 4));
        assert!(m.insert(1, 2));

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
            assert!(m.insert(i * 2, i * 4));
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

        assert!(m.insert(3i, 6i));
        assert!(m.insert(0, 0));
        assert!(m.insert(4, 8));
        assert!(m.insert(2, 4));
        assert!(m.insert(1, 2));

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
            assert!(m.insert(i, 100 * i));
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
            assert!(m.insert(i, 100 * i));
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
            assert!(m_lower.insert(i * 2, i * 4));
            assert!(m_upper.insert(i * 2, i * 4));
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
        assert!(a.insert(0i, 5i));
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
        assert!(b.insert(0i, 5i));
        assert!(a < b);
        assert!(a.insert(0, 7));
        assert!(!(a < b) && b < a);
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
        assert!(a.insert(1i, 1i));
        assert!(a > b && a >= b);
        assert!(b < a && b <= a);
        assert!(b.insert(2, 2));
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

        assert!(m.insert(x1, y1));
        assert!(m.insert(x2, y2));
        assert!(m.insert(x3, y3));
        assert!(m.insert(x4, y4));
        assert!(m.insert(x5, y5));

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
            assert_eq!(map.find(&k), Some(&v));
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
}

#[cfg(test)]
mod bench {
    use std::prelude::*;
    use std::rand::{weak_rng, Rng};
    use test::{Bencher, black_box};

    use super::TreeMap;
    use MutableMap;
    use deque::bench::{insert_rand_n, insert_seq_n, find_rand_n, find_seq_n};

    // Find seq
    #[bench]
    pub fn insert_rand_100(b: &mut Bencher) {
        let mut m : TreeMap<uint,uint> = TreeMap::new();
        insert_rand_n(100, &mut m, b);
    }

    #[bench]
    pub fn insert_rand_10_000(b: &mut Bencher) {
        let mut m : TreeMap<uint,uint> = TreeMap::new();
        insert_rand_n(10_000, &mut m, b);
    }

    // Insert seq
    #[bench]
    pub fn insert_seq_100(b: &mut Bencher) {
        let mut m : TreeMap<uint,uint> = TreeMap::new();
        insert_seq_n(100, &mut m, b);
    }

    #[bench]
    pub fn insert_seq_10_000(b: &mut Bencher) {
        let mut m : TreeMap<uint,uint> = TreeMap::new();
        insert_seq_n(10_000, &mut m, b);
    }

    // Find rand
    #[bench]
    pub fn find_rand_100(b: &mut Bencher) {
        let mut m : TreeMap<uint,uint> = TreeMap::new();
        find_rand_n(100, &mut m, b);
    }

    #[bench]
    pub fn find_rand_10_000(b: &mut Bencher) {
        let mut m : TreeMap<uint,uint> = TreeMap::new();
        find_rand_n(10_000, &mut m, b);
    }

    // Find seq
    #[bench]
    pub fn find_seq_100(b: &mut Bencher) {
        let mut m : TreeMap<uint,uint> = TreeMap::new();
        find_seq_n(100, &mut m, b);
    }

    #[bench]
    pub fn find_seq_10_000(b: &mut Bencher) {
        let mut m : TreeMap<uint,uint> = TreeMap::new();
        find_seq_n(10_000, &mut m, b);
    }

    fn bench_iter(b: &mut Bencher, size: uint) {
        let mut map = TreeMap::<uint, uint>::new();
        let mut rng = weak_rng();

        for _ in range(0, size) {
            map.swap(rng.gen(), rng.gen());
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

#[cfg(test)]
mod test_set {
    use std::prelude::*;
    use std::hash;

    use {Set, MutableSet, Mutable, MutableMap, MutableSeq};
    use super::{TreeMap, TreeSet};

    #[test]
    fn test_clear() {
        let mut s = TreeSet::new();
        s.clear();
        assert!(s.insert(5i));
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
        assert!(xs.insert(5i));
        assert!(ys.insert(11i));
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
        assert!(a.insert(0i));
        assert!(a.insert(5));
        assert!(a.insert(11));
        assert!(a.insert(7));

        let mut b = TreeSet::new();
        assert!(b.insert(0i));
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
    fn test_iterator() {
        let mut m = TreeSet::new();

        assert!(m.insert(3i));
        assert!(m.insert(0));
        assert!(m.insert(4));
        assert!(m.insert(2));
        assert!(m.insert(1));

        let mut n = 0;
        for x in m.iter() {
            assert_eq!(*x, n);
            n += 1
        }
    }

    #[test]
    fn test_rev_iter() {
        let mut m = TreeSet::new();

        assert!(m.insert(3i));
        assert!(m.insert(0));
        assert!(m.insert(4));
        assert!(m.insert(2));
        assert!(m.insert(1));

        let mut n = 4;
        for x in m.rev_iter() {
            assert_eq!(*x, n);
            n -= 1;
        }
    }

    #[test]
    fn test_move_iter() {
        let s: TreeSet<int> = range(0i, 5).collect();

        let mut n = 0;
        for x in s.into_iter() {
            assert_eq!(x, n);
            n += 1;
        }
    }

    #[test]
    fn test_move_iter_size_hint() {
        let s: TreeSet<int> = vec!(0i, 1).into_iter().collect();

        let mut it = s.into_iter();

        assert_eq!(it.size_hint(), (2, Some(2)));
        assert!(it.next() != None);

        assert_eq!(it.size_hint(), (1, Some(1)));
        assert!(it.next() != None);

        assert_eq!(it.size_hint(), (0, Some(0)));
        assert_eq!(it.next(), None);
    }

    #[test]
    fn test_clone_eq() {
      let mut m = TreeSet::new();

      m.insert(1i);
      m.insert(2);

      assert!(m.clone() == m);
    }

    #[test]
    fn test_hash() {
      let mut x = TreeSet::new();
      let mut y = TreeSet::new();

      x.insert(1i);
      x.insert(2);
      x.insert(3);

      y.insert(3i);
      y.insert(2);
      y.insert(1);

      assert!(hash::hash(&x) == hash::hash(&y));
    }

    fn check(a: &[int],
             b: &[int],
             expected: &[int],
             f: |&TreeSet<int>, &TreeSet<int>, f: |&int| -> bool| -> bool) {
        let mut set_a = TreeSet::new();
        let mut set_b = TreeSet::new();

        for x in a.iter() { assert!(set_a.insert(*x)) }
        for y in b.iter() { assert!(set_b.insert(*y)) }

        let mut i = 0;
        f(&set_a, &set_b, |x| {
            assert_eq!(*x, expected[i]);
            i += 1;
            true
        });
        assert_eq!(i, expected.len());
    }

    #[test]
    fn test_intersection() {
        fn check_intersection(a: &[int], b: &[int], expected: &[int]) {
            check(a, b, expected, |x, y, f| x.intersection(y).all(f))
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
            check(a, b, expected, |x, y, f| x.difference(y).all(f))
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
            check(a, b, expected, |x, y, f| x.symmetric_difference(y).all(f))
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
            check(a, b, expected, |x, y, f| x.union(y).all(f))
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
        assert_eq!(result.unwrap(), (&5u, &("bar")));

        let result: Option<(&uint, & &'static str)> = z.next();
        assert_eq!(result.unwrap(), (&11u, &("foo")));

        let result: Option<(&uint, & &'static str)> = z.next();
        assert!(result.is_none());
    }

    #[test]
    fn test_swap() {
        let mut m = TreeMap::new();
        assert_eq!(m.swap(1u, 2i), None);
        assert_eq!(m.swap(1u, 3i), Some(2));
        assert_eq!(m.swap(1u, 4i), Some(3));
    }

    #[test]
    fn test_pop() {
        let mut m = TreeMap::new();
        m.insert(1u, 2i);
        assert_eq!(m.pop(&1), Some(2));
        assert_eq!(m.pop(&1), None);
    }

    #[test]
    fn test_from_iter() {
        let xs = [1i, 2, 3, 4, 5, 6, 7, 8, 9];

        let set: TreeSet<int> = xs.iter().map(|&x| x).collect();

        for x in xs.iter() {
            assert!(set.contains(x));
        }
    }

    #[test]
    fn test_show() {
        let mut set: TreeSet<int> = TreeSet::new();
        let empty: TreeSet<int> = TreeSet::new();

        set.insert(1);
        set.insert(2);

        let set_str = format!("{}", set);

        assert!(set_str == "{1, 2}".to_string());
        assert_eq!(format!("{}", empty), "{}".to_string());
    }
}
