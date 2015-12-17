// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use core::cmp::Ordering;
use core::fmt::Debug;
use core::hash::{Hash, Hasher};
use core::iter::{FromIterator, Map};
use core::ops::Index;
use core::{fmt, intrinsics, mem, ptr};

use borrow::Borrow;
use Bound::{self, Included, Excluded, Unbounded};

use super::node::{self, NodeRef, Handle, marker};
use super::search;

use super::node::InsertResult::*;
use super::node::ForceResult::*;
use super::search::SearchResult::*;
use self::UnderflowResult::*;
use self::Entry::*;

/// A map based on a B-Tree.
///
/// B-Trees represent a fundamental compromise between cache-efficiency and actually minimizing
/// the amount of work performed in a search. In theory, a binary search tree (BST) is the optimal
/// choice for a sorted map, as a perfectly balanced BST performs the theoretical minimum amount of
/// comparisons necessary to find an element (log<sub>2</sub>n). However, in practice the way this
/// is done is *very* inefficient for modern computer architectures. In particular, every element
/// is stored in its own individually heap-allocated node. This means that every single insertion
/// triggers a heap-allocation, and every single comparison should be a cache-miss. Since these
/// are both notably expensive things to do in practice, we are forced to at very least reconsider
/// the BST strategy.
///
/// A B-Tree instead makes each node contain B-1 to 2B-1 elements in a contiguous array. By doing
/// this, we reduce the number of allocations by a factor of B, and improve cache efficiency in
/// searches. However, this does mean that searches will have to do *more* comparisons on average.
/// The precise number of comparisons depends on the node search strategy used. For optimal cache
/// efficiency, one could search the nodes linearly. For optimal comparisons, one could search
/// the node using binary search. As a compromise, one could also perform a linear search
/// that initially only checks every i<sup>th</sup> element for some choice of i.
///
/// Currently, our implementation simply performs naive linear search. This provides excellent
/// performance on *small* nodes of elements which are cheap to compare. However in the future we
/// would like to further explore choosing the optimal search strategy based on the choice of B,
/// and possibly other factors. Using linear search, searching for a random element is expected
/// to take O(B log<sub>B</sub>n) comparisons, which is generally worse than a BST. In practice,
/// however, performance is excellent.
///
/// It is a logic error for a key to be modified in such a way that the key's ordering relative to
/// any other key, as determined by the `Ord` trait, changes while it is in the map. This is
/// normally only possible through `Cell`, `RefCell`, global state, I/O, or unsafe code.
#[stable(feature = "rust1", since = "1.0.0")]
pub struct BTreeMap<K, V> {
    root: node::Root<K, V>,
    length: usize
}

impl<K, V> Drop for BTreeMap<K, V> {
    fn drop(&mut self) {
        unsafe {
            for _ in ptr::read(self).into_iter() { }
        }
    }
}

impl<K: Clone, V: Clone> Clone for BTreeMap<K, V> {
    fn clone(&self) -> BTreeMap<K, V> {
        fn create_chain<K, V>(height: usize) -> node::Root<K, V> {
            let mut ret = node::Root::new_leaf();
            for _ in 0..height {
                ret.enlarge();
            }
            ret
        }

        let mut out_root = create_chain(self.root.as_ref().height());

        {
            let mut in_node = first_leaf_edge(self.root.as_ref()).into_node();
            let mut out_node = first_leaf_edge(out_root.as_mut()).into_node();
            'main: loop {
                // Copy the leaf node
                let mut in_edge = in_node.first_edge();
                while let Ok(kv) = in_edge.right_kv() {
                    let (k, v) = kv.into_kv();
                    out_node.push(k.clone(), v.clone());

                    in_edge = kv.right_edge();
                }

                // Find the next leaf node
                let mut internal_in_edge = match in_node.ascend() {
                    Ok(parent) => parent,
                    Err(_) => break
                };
                let mut internal_out_node = out_node.ascend().ok().unwrap().into_node();
                let internal_kv;
                loop {
                    match internal_in_edge.right_kv() {
                        Ok(kv) => {
                            internal_kv = kv;
                            break;
                        },
                        Err(edge) => {
                            internal_in_edge = match edge.into_node().ascend() {
                                Ok(parent) => parent,
                                Err(_) => break 'main
                            };
                            internal_out_node = internal_out_node.ascend()
                                                                 .ok()
                                                                 .unwrap()
                                                                 .into_node();
                        }
                    }
                }

                let (k, v) = internal_kv.into_kv();
                internal_out_node.push(
                    k.clone(),
                    v.clone(),
                    create_chain(internal_kv.into_node().height() - 1)
                );

                in_node = first_leaf_edge(internal_kv.right_edge().descend()).into_node();
                out_node = first_leaf_edge(internal_out_node.last_edge().descend()).into_node();
            }
        }

        BTreeMap {
            root: out_root,
            length: self.length
        }
    }
}

impl<K, Q: ?Sized> super::Recover<Q> for BTreeMap<K, ()>
    where K: Borrow<Q> + Ord,
          Q: Ord
{
    type Key = K;

    fn get(&self, key: &Q) -> Option<&K> {
        match search::search_tree(self.root.as_ref(), key) {
            Found(handle) => Some(handle.into_kv().0),
            GoDown(_) => None
        }
    }

    fn take(&mut self, key: &Q) -> Option<K> {
        match search::search_tree(self.root.as_mut(), key) {
            Found(handle) => {
                Some(OccupiedEntry {
                    handle: handle,
                    length: &mut self.length
                }.remove_kv().0)
            },
            GoDown(_) => None
        }
    }

    fn replace(&mut self, key: K) -> Option<K> {
        match self.entry(key) {
            Occupied(occupied) => Some(occupied.remove_kv().0),
            Vacant(_) => None
        }
    }
}

/// An iterator over a BTreeMap's entries.
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Iter<'a, K: 'a, V: 'a> {
    range: Range<'a, K, V>,
    length: usize
}

/// A mutable iterator over a BTreeMap's entries.
#[stable(feature = "rust1", since = "1.0.0")]
pub struct IterMut<'a, K: 'a, V: 'a> {
    range: RangeMut<'a, K, V>,
    length: usize
}

/// An owning iterator over a BTreeMap's entries.
#[stable(feature = "rust1", since = "1.0.0")]
pub struct IntoIter<K, V> {
    front: Handle<NodeRef<marker::Owned, K, V, marker::Mut, marker::Leaf>, marker::Edge>,
    back: Handle<NodeRef<marker::Owned, K, V, marker::Mut, marker::Leaf>, marker::Edge>,
    length: usize
}

/// An iterator over a BTreeMap's keys.
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Keys<'a, K: 'a, V: 'a> {
    inner: Map<Iter<'a, K, V>, fn((&'a K, &'a V)) -> &'a K>,
}

/// An iterator over a BTreeMap's values.
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Values<'a, K: 'a, V: 'a> {
    inner: Map<Iter<'a, K, V>, fn((&'a K, &'a V)) -> &'a V>,
}

/// An iterator over a sub-range of BTreeMap's entries.
pub struct Range<'a, K: 'a, V: 'a> {
    front: Handle<NodeRef<marker::Borrowed<'a>, K, V, marker::Immut, marker::Leaf>, marker::Edge>,
    back: Handle<NodeRef<marker::Borrowed<'a>, K, V, marker::Immut, marker::Leaf>, marker::Edge>
}

/// A mutable iterator over a sub-range of BTreeMap's entries.
pub struct RangeMut<'a, K: 'a, V: 'a> {
    front: Handle<NodeRef<marker::Borrowed<'a>, K, V, marker::Mut, marker::Leaf>, marker::Edge>,
    back: Handle<NodeRef<marker::Borrowed<'a>, K, V, marker::Mut, marker::Leaf>, marker::Edge>
}

/// A view into a single entry in a map, which may either be vacant or occupied.
#[stable(feature = "rust1", since = "1.0.0")]
pub enum Entry<'a, K: 'a, V: 'a> {
    /// A vacant Entry
    #[stable(feature = "rust1", since = "1.0.0")]
    Vacant(VacantEntry<'a, K, V>),

    /// An occupied Entry
    #[stable(feature = "rust1", since = "1.0.0")]
    Occupied(OccupiedEntry<'a, K, V>),
}

/// A vacant Entry.
#[stable(feature = "rust1", since = "1.0.0")]
pub struct VacantEntry<'a, K: 'a, V: 'a> {
    key: K,
    handle: Handle<NodeRef<marker::Borrowed<'a>, K, V, marker::Mut, marker::Leaf>, marker::Edge>,
    length: &'a mut usize
}

/// An occupied Entry.
#[stable(feature = "rust1", since = "1.0.0")]
pub struct OccupiedEntry<'a, K: 'a, V: 'a> {
    handle: Handle<NodeRef<
        marker::Borrowed<'a>,
        K, V,
        marker::Mut,
        marker::LeafOrInternal
    >, marker::KV>,

    length: &'a mut usize
}

impl<K: Ord, V> BTreeMap<K, V> {
    /// Makes a new empty BTreeMap with a reasonable choice for B.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn new() -> BTreeMap<K, V> {
        BTreeMap {
            root: node::Root::new_leaf(),
            length: 0
        }
    }

    /// Clears the map, removing all values.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::BTreeMap;
    ///
    /// let mut a = BTreeMap::new();
    /// a.insert(1, "a");
    /// a.clear();
    /// assert!(a.is_empty());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn clear(&mut self) {
        *self = BTreeMap::new();
    }

    /// Returns a reference to the value corresponding to the key.
    ///
    /// The key may be any borrowed form of the map's key type, but the ordering
    /// on the borrowed form *must* match the ordering on the key type.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::BTreeMap;
    ///
    /// let mut map = BTreeMap::new();
    /// map.insert(1, "a");
    /// assert_eq!(map.get(&1), Some(&"a"));
    /// assert_eq!(map.get(&2), None);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn get<Q: ?Sized>(&self, key: &Q) -> Option<&V> where K: Borrow<Q>, Q: Ord {
        match search::search_tree(self.root.as_ref(), key) {
            Found(handle) => Some(handle.into_kv().1),
            GoDown(_) => None
        }
    }

    /// Returns true if the map contains a value for the specified key.
    ///
    /// The key may be any borrowed form of the map's key type, but the ordering
    /// on the borrowed form *must* match the ordering on the key type.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::BTreeMap;
    ///
    /// let mut map = BTreeMap::new();
    /// map.insert(1, "a");
    /// assert_eq!(map.contains_key(&1), true);
    /// assert_eq!(map.contains_key(&2), false);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn contains_key<Q: ?Sized>(&self, key: &Q) -> bool where K: Borrow<Q>, Q: Ord {
        self.get(key).is_some()
    }

    /// Returns a mutable reference to the value corresponding to the key.
    ///
    /// The key may be any borrowed form of the map's key type, but the ordering
    /// on the borrowed form *must* match the ordering on the key type.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::BTreeMap;
    ///
    /// let mut map = BTreeMap::new();
    /// map.insert(1, "a");
    /// if let Some(x) = map.get_mut(&1) {
    ///     *x = "b";
    /// }
    /// assert_eq!(map[&1], "b");
    /// ```
    // See `get` for implementation notes, this is basically a copy-paste with mut's added
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn get_mut<Q: ?Sized>(&mut self, key: &Q) -> Option<&mut V> where K: Borrow<Q>, Q: Ord {
        match search::search_tree(self.root.as_mut(), key) {
            Found(handle) => Some(handle.into_kv_mut().1),
            GoDown(_) => None
        }
    }

    /// Inserts a key-value pair into the map.
    ///
    /// If the map did not have this key present, `None` is returned.
    ///
    /// If the map did have this key present, the key is not updated, the
    /// value is updated and the old value is returned.
    /// See the [module-level documentation] for more.
    ///
    /// [module-level documentation]: index.html#insert-and-complex-keys
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::BTreeMap;
    ///
    /// let mut map = BTreeMap::new();
    /// assert_eq!(map.insert(37, "a"), None);
    /// assert_eq!(map.is_empty(), false);
    ///
    /// map.insert(37, "b");
    /// assert_eq!(map.insert(37, "c"), Some("b"));
    /// assert_eq!(map[&37], "c");
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        match self.entry(key) {
            Occupied(mut entry) => Some(entry.insert(value)),
            Vacant(entry) => {
                entry.insert(value);
                None
            }
        }
    }

    /// Removes a key from the map, returning the value at the key if the key
    /// was previously in the map.
    ///
    /// The key may be any borrowed form of the map's key type, but the ordering
    /// on the borrowed form *must* match the ordering on the key type.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::BTreeMap;
    ///
    /// let mut map = BTreeMap::new();
    /// map.insert(1, "a");
    /// assert_eq!(map.remove(&1), Some("a"));
    /// assert_eq!(map.remove(&1), None);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn remove<Q: ?Sized>(&mut self, key: &Q) -> Option<V> where K: Borrow<Q>, Q: Ord {
        match search::search_tree(self.root.as_mut(), key) {
            Found(handle) => {
                Some(OccupiedEntry {
                    handle: handle,
                    length: &mut self.length
                }.remove())
            },
            GoDown(_) => None
        }
    }

    /// Constructs a double-ended iterator over a sub-range of elements in the map, starting
    /// at min, and ending at max. If min is `Unbounded`, then it will be treated as "negative
    /// infinity", and if max is `Unbounded`, then it will be treated as "positive infinity".
    /// Thus range(Unbounded, Unbounded) will yield the whole collection.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(btree_range, collections_bound)]
    ///
    /// use std::collections::BTreeMap;
    /// use std::collections::Bound::{Included, Unbounded};
    ///
    /// let mut map = BTreeMap::new();
    /// map.insert(3, "a");
    /// map.insert(5, "b");
    /// map.insert(8, "c");
    /// for (&key, &value) in map.range(Included(&4), Included(&8)) {
    ///     println!("{}: {}", key, value);
    /// }
    /// assert_eq!(Some((&5, &"b")), map.range(Included(&4), Unbounded).next());
    /// ```
    #[unstable(feature = "btree_range",
               reason = "matches collection reform specification, waiting for dust to settle",
               issue = "27787")]
    pub fn range<Min: ?Sized + Ord = K, Max: ?Sized + Ord = K>(&self,
                                                               min: Bound<&Min>,
                                                               max: Bound<&Max>)
                                                               -> Range<K, V>
        where K: Borrow<Min> + Borrow<Max>,
    {
        let front = match min {
            Included(key) => match search::search_tree(self.root.as_ref(), key) {
                Found(kv_handle) => match kv_handle.left_edge().force() {
                    Leaf(bottom) => bottom,
                    Internal(internal) => last_leaf_edge(internal.descend())
                },
                GoDown(bottom) => bottom
            },
            Excluded(key) => match search::search_tree(self.root.as_ref(), key) {
                Found(kv_handle) => match kv_handle.right_edge().force() {
                    Leaf(bottom) => bottom,
                    Internal(internal) => first_leaf_edge(internal.descend())
                },
                GoDown(bottom) => bottom
            },
            Unbounded => first_leaf_edge(self.root.as_ref())
        };

        let back = match max {
            Included(key) => match search::search_tree(self.root.as_ref(), key) {
                Found(kv_handle) => match kv_handle.right_edge().force() {
                    Leaf(bottom) => bottom,
                    Internal(internal) => first_leaf_edge(internal.descend())
                },
                GoDown(bottom) => bottom
            },
            Excluded(key) => match search::search_tree(self.root.as_ref(), key) {
                Found(kv_handle) => match kv_handle.left_edge().force() {
                    Leaf(bottom) => bottom,
                    Internal(internal) => last_leaf_edge(internal.descend())
                },
                GoDown(bottom) => bottom
            },
            Unbounded => last_leaf_edge(self.root.as_ref())
        };

        Range {
            front: front,
            back: back
        }
    }

    /// Constructs a mutable double-ended iterator over a sub-range of elements in the map, starting
    /// at min, and ending at max. If min is `Unbounded`, then it will be treated as "negative
    /// infinity", and if max is `Unbounded`, then it will be treated as "positive infinity".
    /// Thus range(Unbounded, Unbounded) will yield the whole collection.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(btree_range, collections_bound)]
    ///
    /// use std::collections::BTreeMap;
    /// use std::collections::Bound::{Included, Excluded};
    ///
    /// let mut map: BTreeMap<&str, i32> = ["Alice", "Bob", "Carol", "Cheryl"].iter()
    ///                                                                       .map(|&s| (s, 0))
    ///                                                                       .collect();
    /// for (_, balance) in map.range_mut(Included("B"), Excluded("Cheryl")) {
    ///     *balance += 100;
    /// }
    /// for (name, balance) in &map {
    ///     println!("{} => {}", name, balance);
    /// }
    /// ```
    #[unstable(feature = "btree_range",
               reason = "matches collection reform specification, waiting for dust to settle",
               issue = "27787")]
    pub fn range_mut<Min: ?Sized + Ord = K, Max: ?Sized + Ord = K>(&mut self,
                                                                   min: Bound<&Min>,
                                                                   max: Bound<&Max>)
                                                                   -> RangeMut<K, V>
        where K: Borrow<Min> + Borrow<Max>,
    {
        let root1 = self.root.as_mut();
        let root2 = unsafe { ptr::read(&root1) };

        let front = match min {
            Included(key) => match search::search_tree(root1, key) {
                Found(kv_handle) => match kv_handle.left_edge().force() {
                    Leaf(bottom) => bottom,
                    Internal(internal) => last_leaf_edge(internal.descend())
                },
                GoDown(bottom) => bottom
            },
            Excluded(key) => match search::search_tree(root1, key) {
                Found(kv_handle) => match kv_handle.right_edge().force() {
                    Leaf(bottom) => bottom,
                    Internal(internal) => first_leaf_edge(internal.descend())
                },
                GoDown(bottom) => bottom
            },
            Unbounded => first_leaf_edge(root1)
        };

        let back = match max {
            Included(key) => match search::search_tree(root2, key) {
                Found(kv_handle) => match kv_handle.right_edge().force() {
                    Leaf(bottom) => bottom,
                    Internal(internal) => first_leaf_edge(internal.descend())
                },
                GoDown(bottom) => bottom
            },
            Excluded(key) => match search::search_tree(root2, key) {
                Found(kv_handle) => match kv_handle.left_edge().force() {
                    Leaf(bottom) => bottom,
                    Internal(internal) => last_leaf_edge(internal.descend())
                },
                GoDown(bottom) => bottom
            },
            Unbounded => last_leaf_edge(root2)
        };

        RangeMut {
            front: front,
            back: back
        }
    }

    /// Gets the given key's corresponding entry in the map for in-place manipulation.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::BTreeMap;
    ///
    /// let mut count: BTreeMap<&str, usize> = BTreeMap::new();
    ///
    /// // count the number of occurrences of letters in the vec
    /// for x in vec!["a","b","a","c","a","b"] {
    ///     *count.entry(x).or_insert(0) += 1;
    /// }
    ///
    /// assert_eq!(count["a"], 3);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn entry(&mut self, key: K) -> Entry<K, V> {
        match search::search_tree(self.root.as_mut(), &key) {
            Found(handle) => Occupied(OccupiedEntry {
                handle: handle,
                length: &mut self.length
            }),
            GoDown(handle) => Vacant(VacantEntry {
                key: key,
                handle: handle,
                length: &mut self.length
            })
        }
    }
}

impl<'a, K: 'a, V: 'a> IntoIterator for &'a BTreeMap<K, V> {
    type Item = (&'a K, &'a V);
    type IntoIter = Iter<'a, K, V>;

    fn into_iter(self) -> Iter<'a, K, V> {
        self.iter()
    }
}

impl<'a, K: 'a, V: 'a> Iterator for Iter<'a, K, V> {
    type Item = (&'a K, &'a V);

    fn next(&mut self) -> Option<(&'a K, &'a V)> {
        if self.length == 0 {
            None
        } else {
            self.length -= 1;
            unsafe { Some(self.range.next_unchecked()) }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.length, Some(self.length))
    }
}

impl<'a, K: 'a, V: 'a> DoubleEndedIterator for Iter<'a, K, V> {
    fn next_back(&mut self) -> Option<(&'a K, &'a V)> {
        if self.length == 0 {
            None
        } else {
            self.length -= 1;
            unsafe { Some(self.range.next_back_unchecked()) }
        }
    }
}

impl<'a, K: 'a, V: 'a> ExactSizeIterator for Iter<'a, K, V> {
    fn len(&self) -> usize { self.length }
}

impl<'a, K, V> Clone for Iter<'a, K, V> {
    fn clone(&self) -> Iter<'a, K, V> {
        Iter {
            range: self.range.clone(),
            length: self.length
        }
    }
}

impl<'a, K: 'a, V: 'a> IntoIterator for &'a mut BTreeMap<K, V> {
    type Item = (&'a K, &'a mut V);
    type IntoIter = IterMut<'a, K, V>;

    fn into_iter(self) -> IterMut<'a, K, V> {
        self.iter_mut()
    }
}

impl<'a, K: 'a, V: 'a> Iterator for IterMut<'a, K, V> {
    type Item = (&'a K, &'a mut V);

    fn next(&mut self) -> Option<(&'a K, &'a mut V)> {
        if self.length == 0 {
            None
        } else {
            self.length -= 1;
            unsafe { Some(self.range.next_unchecked()) }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.length, Some(self.length))
    }
}

impl<'a, K: 'a, V: 'a> DoubleEndedIterator for IterMut<'a, K, V> {
    fn next_back(&mut self) -> Option<(&'a K, &'a mut V)> {
        if self.length == 0 {
            None
        } else {
            self.length -= 1;
            unsafe { Some(self.range.next_back_unchecked()) }
        }
    }
}

impl<'a, K: 'a, V: 'a> ExactSizeIterator for IterMut<'a, K, V> {
    fn len(&self) -> usize { self.length }
}

impl<K, V> IntoIterator for BTreeMap<K, V> {
    type Item = (K, V);
    type IntoIter = IntoIter<K, V>;

    fn into_iter(self) -> IntoIter<K, V> {
        let root1 = unsafe { ptr::read(&self.root).into_ref() };
        let root2 = unsafe { ptr::read(&self.root).into_ref() };
        let len = self.length;
        mem::forget(self);

        IntoIter {
            front: first_leaf_edge(root1),
            back: last_leaf_edge(root2),
            length: len
        }
    }
}

impl<K, V> Drop for IntoIter<K, V> {
    fn drop(&mut self) {
        for _ in &mut *self { }
        unsafe {
            let leaf_node = ptr::read(&self.front).into_node();
            if let Some(first_parent) = leaf_node.deallocate_and_ascend() {
                let mut cur_node = first_parent.into_node();
                while let Some(parent) = cur_node.deallocate_and_ascend() {
                    cur_node = parent.into_node()
                }
            }
        }
    }
}

impl<K, V> Iterator for IntoIter<K, V> {
    type Item = (K, V);

    fn next(&mut self) -> Option<(K, V)> {
        if self.length == 0 {
            return None;
        } else {
            self.length -= 1;
        }

        let handle = unsafe { ptr::read(&self.front) };

        let mut cur_handle = match handle.right_kv() {
            Ok(kv) => {
                let k = unsafe { ptr::read(kv.reborrow().into_kv().0) };
                let v = unsafe { ptr::read(kv.reborrow().into_kv().1) };
                self.front = kv.right_edge();
                return Some((k, v));
            },
            Err(last_edge) => unsafe {
                unwrap_unchecked(last_edge.into_node().deallocate_and_ascend())
            }
        };

        loop {
            match cur_handle.right_kv() {
                Ok(kv) => {
                    let k = unsafe { ptr::read(kv.reborrow().into_kv().0) };
                    let v = unsafe { ptr::read(kv.reborrow().into_kv().1) };
                    self.front = first_leaf_edge(kv.right_edge().descend());
                    return Some((k, v));
                },
                Err(last_edge) => unsafe {
                    cur_handle = unwrap_unchecked(last_edge.into_node().deallocate_and_ascend());
                }
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.length, Some(self.length))
    }
}

impl<K, V> DoubleEndedIterator for IntoIter<K, V> {
    fn next_back(&mut self) -> Option<(K, V)> {
        if self.length == 0 {
            return None;
        } else {
            self.length -= 1;
        }

        let handle = unsafe { ptr::read(&self.back) };

        let mut cur_handle = match handle.left_kv() {
            Ok(kv) => {
                let k = unsafe { ptr::read(kv.reborrow().into_kv().0) };
                let v = unsafe { ptr::read(kv.reborrow().into_kv().1) };
                self.back = kv.left_edge();
                return Some((k, v));
            },
            Err(last_edge) => unsafe {
                unwrap_unchecked(last_edge.into_node().deallocate_and_ascend())
            }
        };

        loop {
            match cur_handle.left_kv() {
                Ok(kv) => {
                    let k = unsafe { ptr::read(kv.reborrow().into_kv().0) };
                    let v = unsafe { ptr::read(kv.reborrow().into_kv().1) };
                    self.back = last_leaf_edge(kv.left_edge().descend());
                    return Some((k, v));
                },
                Err(last_edge) => unsafe {
                    cur_handle = unwrap_unchecked(last_edge.into_node().deallocate_and_ascend());
                }
            }
        }
    }
}

impl<K, V> ExactSizeIterator for IntoIter<K, V> {
    fn len(&self) -> usize { self.length }
}

impl<'a, K, V> Iterator for Keys<'a, K, V> {
    type Item = &'a K;

    fn next(&mut self) -> Option<&'a K> {
        self.inner.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<'a, K, V> DoubleEndedIterator for Keys<'a, K, V> {
    fn next_back(&mut self) -> Option<&'a K> {
        self.inner.next_back()
    }
}

impl<'a, K, V> ExactSizeIterator for Keys<'a, K, V> {
    fn len(&self) -> usize {
        self.inner.len()
    }
}

impl<'a, K, V> Clone for Keys<'a, K, V> {
    fn clone(&self) -> Keys<'a, K, V> {
        Keys {
            inner: self.inner.clone()
        }
    }
}

impl<'a, K, V> Iterator for Values<'a, K, V> {
    type Item = &'a V;

    fn next(&mut self) -> Option<&'a V> {
        self.inner.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<'a, K, V> DoubleEndedIterator for Values<'a, K, V> {
    fn next_back(&mut self) -> Option<&'a V> {
        self.inner.next_back()
    }
}

impl<'a, K, V> ExactSizeIterator for Values<'a, K, V> {
    fn len(&self) -> usize {
        self.inner.len()
    }
}

impl<'a, K, V> Clone for Values<'a, K, V> {
    fn clone(&self) -> Values<'a, K, V> {
        Values {
            inner: self.inner.clone()
        }
    }
}

impl<'a, K, V> Iterator for Range<'a, K, V> {
    type Item = (&'a K, &'a V);

    fn next(&mut self) -> Option<(&'a K, &'a V)> {
        if self.front == self.back {
            None
        } else {
            unsafe { Some(self.next_unchecked()) }
        }
    }
}

impl<'a, K, V> Range<'a, K, V> {
    unsafe fn next_unchecked(&mut self) -> (&'a K, &'a V) {
        let handle = self.front;

        let mut cur_handle = match handle.right_kv() {
            Ok(kv) => {
                let ret = kv.into_kv();
                self.front = kv.right_edge();
                return ret;
            },
            Err(last_edge) => {
                let next_level = last_edge.into_node().ascend().ok();
                unwrap_unchecked(next_level)
            }
        };

        loop {
            match cur_handle.right_kv() {
                Ok(kv) => {
                    let ret = kv.into_kv();
                    self.front = first_leaf_edge(kv.right_edge().descend());
                    return ret;
                },
                Err(last_edge) => {
                    let next_level = last_edge.into_node().ascend().ok();
                    cur_handle = unwrap_unchecked(next_level);
                }
            }
        }
    }
}

impl<'a, K, V> DoubleEndedIterator for Range<'a, K, V> {
    fn next_back(&mut self) -> Option<(&'a K, &'a V)> {
        if self.front == self.back {
            None
        } else {
            unsafe { Some(self.next_back_unchecked()) }
        }
    }
}

impl<'a, K, V> Range<'a, K, V> {
    unsafe fn next_back_unchecked(&mut self) -> (&'a K, &'a V) {
        let handle = self.back;

        let mut cur_handle = match handle.left_kv() {
            Ok(kv) => {
                let ret = kv.into_kv();
                self.back = kv.left_edge();
                return ret;
            },
            Err(last_edge) => {
                let next_level = last_edge.into_node().ascend().ok();
                unwrap_unchecked(next_level)
            }
        };

        loop {
            match cur_handle.left_kv() {
                Ok(kv) => {
                    let ret = kv.into_kv();
                    self.back = last_leaf_edge(kv.left_edge().descend());
                    return ret;
                },
                Err(last_edge) => {
                    let next_level = last_edge.into_node().ascend().ok();
                    cur_handle = unwrap_unchecked(next_level);
                }
            }
        }
    }
}

impl<'a, K, V> Clone for Range<'a, K, V> {
    fn clone(&self) -> Range<'a, K, V> {
        Range {
            front: self.front,
            back: self.back
        }
    }
}

impl<'a, K, V> Iterator for RangeMut<'a, K, V> {
    type Item = (&'a K, &'a mut V);

    fn next(&mut self) -> Option<(&'a K, &'a mut V)> {
        if self.front == self.back {
            None
        } else {
            unsafe { Some (self.next_unchecked()) }
        }
    }
}

impl<'a, K, V> RangeMut<'a, K, V> {
    unsafe fn next_unchecked(&mut self) -> (&'a K, &'a mut V) {
        let handle = ptr::read(&self.front);

        let mut cur_handle = match handle.right_kv() {
            Ok(kv) => {
                let (k, v) = ptr::read(&kv).into_kv_mut();
                self.front = kv.right_edge();
                return (k, v);
            },
            Err(last_edge) => {
                let next_level = last_edge.into_node().ascend().ok();
                unwrap_unchecked(next_level)
            }
        };

        loop {
            match cur_handle.right_kv() {
                Ok(kv) => {
                    let (k, v) = ptr::read(&kv).into_kv_mut();
                    self.front = first_leaf_edge(kv.right_edge().descend());
                    return (k, v);
                },
                Err(last_edge) => {
                    let next_level = last_edge.into_node().ascend().ok();
                    cur_handle = unwrap_unchecked(next_level);
                }
            }
        }
    }
}

impl<'a, K, V> DoubleEndedIterator for RangeMut<'a, K, V> {
    fn next_back(&mut self) -> Option<(&'a K, &'a mut V)> {
        if self.front == self.back {
            None
        } else {
            unsafe { Some(self.next_back_unchecked()) }
        }
    }
}

impl<'a, K, V> RangeMut<'a, K, V> {
    unsafe fn next_back_unchecked(&mut self) -> (&'a K, &'a mut V) {
        let handle = ptr::read(&self.back);

        let mut cur_handle = match handle.left_kv() {
            Ok(kv) => {
                let (k, v) = ptr::read(&kv).into_kv_mut();
                self.back = kv.left_edge();
                return (k, v);
            },
            Err(last_edge) => {
                let next_level = last_edge.into_node().ascend().ok();
                unwrap_unchecked(next_level)
            }
        };

        loop {
            match cur_handle.left_kv() {
                Ok(kv) => {
                    let (k, v) = ptr::read(&kv).into_kv_mut();
                    self.back = last_leaf_edge(kv.left_edge().descend());
                    return (k, v);
                },
                Err(last_edge) => {
                    let next_level = last_edge.into_node().ascend().ok();
                    cur_handle = unwrap_unchecked(next_level);
                }
            }
        }
    }
}

impl<K: Ord, V> FromIterator<(K, V)> for BTreeMap<K, V> {
    fn from_iter<T: IntoIterator<Item=(K, V)>>(iter: T) -> BTreeMap<K, V> {
        let mut map = BTreeMap::new();
        map.extend(iter);
        map
    }
}

impl<K: Ord, V> Extend<(K, V)> for BTreeMap<K, V> {
    #[inline]
    fn extend<T: IntoIterator<Item=(K, V)>>(&mut self, iter: T) {
        for (k, v) in iter {
            self.insert(k, v);
        }
    }
}

impl<'a, K: Ord + Copy, V: Copy> Extend<(&'a K, &'a V)> for BTreeMap<K, V> {
    fn extend<I: IntoIterator<Item=(&'a K, &'a V)>>(&mut self, iter: I) {
        self.extend(iter.into_iter().map(|(&key, &value)| (key, value)));
    }
}

impl<K: Hash, V: Hash> Hash for BTreeMap<K, V> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for elt in self {
            elt.hash(state);
        }
    }
}

impl<K: Ord, V> Default for BTreeMap<K, V> {
    fn default() -> BTreeMap<K, V> {
        BTreeMap::new()
    }
}

impl<K: PartialEq, V: PartialEq> PartialEq for BTreeMap<K, V> {
    fn eq(&self, other: &BTreeMap<K, V>) -> bool {
        self.len() == other.len() &&
            self.iter().zip(other).all(|(a, b)| a == b)
    }
}

impl<K: Eq, V: Eq> Eq for BTreeMap<K, V> {}

impl<K: PartialOrd, V: PartialOrd> PartialOrd for BTreeMap<K, V> {
    #[inline]
    fn partial_cmp(&self, other: &BTreeMap<K, V>) -> Option<Ordering> {
        self.iter().partial_cmp(other.iter())
    }
}

impl<K: Ord, V: Ord> Ord for BTreeMap<K, V> {
    #[inline]
    fn cmp(&self, other: &BTreeMap<K, V>) -> Ordering {
        self.iter().cmp(other.iter())
    }
}

impl<K: Debug, V: Debug> Debug for BTreeMap<K, V> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_map().entries(self.iter()).finish()
    }
}

impl<'a, K: Ord, Q: ?Sized, V> Index<&'a Q> for BTreeMap<K, V>
    where K: Borrow<Q>, Q: Ord
{
    type Output = V;

    #[inline]
    fn index(&self, key: &Q) -> &V {
        self.get(key).expect("no entry found for key")
    }
}

fn first_leaf_edge<Lifetime, K, V, Mutability>(
        mut node: NodeRef<Lifetime,
                          K, V,
                          Mutability,
                          marker::LeafOrInternal>
        ) -> Handle<NodeRef<Lifetime, K, V, Mutability, marker::Leaf>, marker::Edge> {
    loop {
        match node.force() {
            Leaf(leaf) => return leaf.first_edge(),
            Internal(internal) => {
                node = internal.first_edge().descend();
            }
        }
    }
}

fn last_leaf_edge<Lifetime, K, V, Mutability>(
        mut node: NodeRef<Lifetime,
                          K, V,
                          Mutability,
                          marker::LeafOrInternal>
        ) -> Handle<NodeRef<Lifetime, K, V, Mutability, marker::Leaf>, marker::Edge> {
    loop {
        match node.force() {
            Leaf(leaf) => return leaf.last_edge(),
            Internal(internal) => {
                node = internal.last_edge().descend();
            }
        }
    }
}

#[inline(always)]
unsafe fn unwrap_unchecked<T>(val: Option<T>) -> T {
    val.unwrap_or_else(|| {
        if cfg!(debug_assertions) {
            panic!("'unchecked' unwrap on None in BTreeMap");
        } else {
            intrinsics::unreachable();
        }
    })
}

impl<K, V> BTreeMap<K, V> {
    /// Gets an iterator over the entries of the map.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::BTreeMap;
    ///
    /// let mut map = BTreeMap::new();
    /// map.insert(1, "a");
    /// map.insert(2, "b");
    /// map.insert(3, "c");
    ///
    /// for (key, value) in map.iter() {
    ///     println!("{}: {}", key, value);
    /// }
    ///
    /// let (first_key, first_value) = map.iter().next().unwrap();
    /// assert_eq!((*first_key, *first_value), (1, "a"));
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn iter(&self) -> Iter<K, V> {
        Iter {
            range: Range {
                front: first_leaf_edge(self.root.as_ref()),
                back: last_leaf_edge(self.root.as_ref())
            },
            length: self.length
        }
    }

    /// Gets a mutable iterator over the entries of the map.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::BTreeMap;
    ///
    /// let mut map = BTreeMap::new();
    /// map.insert("a", 1);
    /// map.insert("b", 2);
    /// map.insert("c", 3);
    ///
    /// // add 10 to the value if the key isn't "a"
    /// for (key, value) in map.iter_mut() {
    ///     if key != &"a" {
    ///         *value += 10;
    ///     }
    /// }
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn iter_mut(&mut self) -> IterMut<K, V> {
        let root1 = self.root.as_mut();
        let root2 = unsafe { ptr::read(&root1) };
        IterMut {
            range: RangeMut {
                front: first_leaf_edge(root1),
                back: last_leaf_edge(root2),
            },
            length: self.length
        }
    }

    /// Gets an iterator over the keys of the map.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::BTreeMap;
    ///
    /// let mut a = BTreeMap::new();
    /// a.insert(1, "a");
    /// a.insert(2, "b");
    ///
    /// let keys: Vec<_> = a.keys().cloned().collect();
    /// assert_eq!(keys, [1, 2]);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn keys<'a>(&'a self) -> Keys<'a, K, V> {
        fn first<A, B>((a, _): (A, B)) -> A {
            a
        }
        let first: fn((&'a K, &'a V)) -> &'a K = first; // coerce to fn pointer

        Keys { inner: self.iter().map(first) }
    }

    /// Gets an iterator over the values of the map.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::BTreeMap;
    ///
    /// let mut a = BTreeMap::new();
    /// a.insert(1, "a");
    /// a.insert(2, "b");
    ///
    /// let values: Vec<&str> = a.values().cloned().collect();
    /// assert_eq!(values, ["a", "b"]);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn values<'a>(&'a self) -> Values<'a, K, V> {
        fn second<A, B>((_, b): (A, B)) -> B {
            b
        }
        let second: fn((&'a K, &'a V)) -> &'a V = second; // coerce to fn pointer

        Values { inner: self.iter().map(second) }
    }

    /// Returns the number of elements in the map.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::BTreeMap;
    ///
    /// let mut a = BTreeMap::new();
    /// assert_eq!(a.len(), 0);
    /// a.insert(1, "a");
    /// assert_eq!(a.len(), 1);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn len(&self) -> usize {
        self.length
    }

    /// Returns true if the map contains no elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::BTreeMap;
    ///
    /// let mut a = BTreeMap::new();
    /// assert!(a.is_empty());
    /// a.insert(1, "a");
    /// assert!(!a.is_empty());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<'a, K: Ord, V> Entry<'a, K, V> {
    /// Ensures a value is in the entry by inserting the default if empty, and returns
    /// a mutable reference to the value in the entry.
    pub fn or_insert(self, default: V) -> &'a mut V {
        match self {
            Occupied(entry) => entry.into_mut(),
            Vacant(entry) => entry.insert(default),
        }
    }

    /// Ensures a value is in the entry by inserting the result of the default function if empty,
    /// and returns a mutable reference to the value in the entry.
    pub fn or_insert_with<F: FnOnce() -> V>(self, default: F) -> &'a mut V {
        match self {
            Occupied(entry) => entry.into_mut(),
            Vacant(entry) => entry.insert(default()),
        }
    }
}

impl<'a, K: Ord, V> VacantEntry<'a, K, V> {
    /// Sets the value of the entry with the VacantEntry's key,
    /// and returns a mutable reference to it.
    pub fn insert(self, value: V) -> &'a mut V {
        *self.length += 1;

        let out_ptr;

        let mut ins_k;
        let mut ins_v;
        let mut ins_edge;

        let mut cur_parent = match self.handle.insert(self.key, value) {
            (Fit(handle), _) => return handle.into_kv_mut().1,
            (Split(left, k, v, right), ptr) => {
                ins_k = k;
                ins_v = v;
                ins_edge = right;
                out_ptr = ptr;
                left.ascend().map_err(|n| n.into_root_mut())
            }
        };

        loop {
            match cur_parent {
                Ok(parent) => match parent.insert(ins_k, ins_v, ins_edge) {
                    Fit(_) => return unsafe { &mut *out_ptr },
                    Split(left, k, v, right) => {
                        ins_k = k;
                        ins_v = v;
                        ins_edge = right;
                        cur_parent = left.ascend().map_err(|n| n.into_root_mut());
                    }
                },
                Err(root) => {
                    root.enlarge().push(ins_k, ins_v, ins_edge);
                    return unsafe { &mut *out_ptr };
                }
            }
        }
    }
}

impl<'a, K: Ord, V> OccupiedEntry<'a, K, V> {
    /// Gets a reference to the value in the entry.
    pub fn get(&self) -> &V {
        self.handle.reborrow().into_kv().1
    }

    /// Gets a mutable reference to the value in the entry.
    pub fn get_mut(&mut self) -> &mut V {
        self.handle.kv_mut().1
    }

    /// Converts the entry into a mutable reference to its value.
    pub fn into_mut(self) -> &'a mut V {
        self.handle.into_kv_mut().1
    }

    /// Sets the value of the entry with the OccupiedEntry's key,
    /// and returns the entry's old value.
    pub fn insert(&mut self, value: V) -> V {
        mem::replace(self.get_mut(), value)
    }

    /// Takes the value of the entry out of the map, and returns it.
    pub fn remove(self) -> V {
        self.remove_kv().1
    }

    fn remove_kv(self) -> (K, V) {
        *self.length -= 1;

        let (small_leaf, old_key, old_val) = match self.handle.force() {
            Leaf(leaf) => {
                let (hole, old_key, old_val) = leaf.remove();
                (hole.into_node(), old_key, old_val)
            },
            Internal(mut internal) => {
                let key_loc = internal.kv_mut().0 as *mut K;
                let val_loc = internal.kv_mut().1 as *mut V;

                let to_remove = first_leaf_edge(internal.right_edge().descend()).right_kv().ok();
                let to_remove = unsafe { unwrap_unchecked(to_remove) };

                let (hole, key, val) = to_remove.remove();

                let old_key = unsafe {
                    mem::replace(&mut *key_loc, key)
                };
                let old_val = unsafe {
                    mem::replace(&mut *val_loc, val)
                };

                (hole.into_node(), old_key, old_val)
            }
        };

        // Handle underflow
        let mut cur_node = small_leaf.forget_type();
        while cur_node.len() < cur_node.capacity() / 2 {
            match handle_underfull_node(cur_node) {
                AtRoot => break,
                EmptyParent(_) => unreachable!(),
                Merged(parent) => if parent.len() == 0 {
                    // We must be at the root
                    parent.into_root_mut().shrink();
                    break;
                } else {
                    cur_node = parent.forget_type();
                },
                Stole(_) => break
            }
        }

        (old_key, old_val)
    }
}

enum UnderflowResult<'a, K, V> {
    AtRoot,
    EmptyParent(NodeRef<marker::Borrowed<'a>, K, V, marker::Mut, marker::Internal>),
    Merged(NodeRef<marker::Borrowed<'a>, K, V, marker::Mut, marker::Internal>),
    Stole(NodeRef<marker::Borrowed<'a>, K, V, marker::Mut, marker::Internal>)
}

fn handle_underfull_node<'a, K, V>(node: NodeRef<marker::Borrowed<'a>,
                                                 K, V,
                                                 marker::Mut,
                                                 marker::LeafOrInternal>)
                                                 -> UnderflowResult<'a, K, V> {
    let parent = if let Ok(parent) = node.ascend() {
        parent
    } else {
        return AtRoot;
    };

    let (is_left, mut handle) = match parent.left_kv() {
        Ok(left) => (true, left),
        Err(parent) => match parent.right_kv() {
            Ok(right) => (false, right),
            Err(parent) => {
                return EmptyParent(parent.into_node());
            }
        }
    };

    if handle.can_merge() {
        return Merged(handle.merge().into_node());
    } else {
        unsafe {
            let (k, v, edge) = if is_left {
                handle.reborrow_mut().left_edge().descend().pop()
            } else {
                handle.reborrow_mut().right_edge().descend().pop_front()
            };

            let k = mem::replace(handle.reborrow_mut().into_kv_mut().0, k);
            let v = mem::replace(handle.reborrow_mut().into_kv_mut().1, v);

            // FIXME: reuse cur_node?
            if is_left {
                match handle.reborrow_mut().right_edge().descend().force() {
                    Leaf(mut leaf) => leaf.push_front(k, v),
                    Internal(mut internal) => internal.push_front(k, v, edge.unwrap())
                }
            } else {
                match handle.reborrow_mut().left_edge().descend().force() {
                    Leaf(mut leaf) => leaf.push(k, v),
                    Internal(mut internal) => internal.push(k, v, edge.unwrap())
                }
            }
        }

        return Stole(handle.into_node());
    }
}
