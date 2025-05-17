use core::borrow::Borrow;
use core::cmp::Ordering;
use core::error::Error;
use core::fmt::{self, Debug};
use core::hash::{Hash, Hasher};
use core::iter::FusedIterator;
use core::marker::PhantomData;
use core::mem::{self, ManuallyDrop};
use core::ops::{Bound, Index, RangeBounds};
use core::ptr;

use super::borrow::DormantMutRef;
use super::dedup_sorted_iter::DedupSortedIter;
use super::navigate::{LazyLeafRange, LeafRange};
use super::node::ForceResult::*;
use super::node::{self, Handle, NodeRef, Root, marker};
use super::search::SearchBound;
use super::search::SearchResult::*;
use super::set_val::SetValZST;
use crate::alloc::{Allocator, Global};
use crate::vec::Vec;

mod entry;

use Entry::*;
#[stable(feature = "rust1", since = "1.0.0")]
pub use entry::{Entry, OccupiedEntry, OccupiedError, VacantEntry};

/// Minimum number of elements in a node that is not a root.
/// We might temporarily have fewer elements during methods.
pub(super) const MIN_LEN: usize = node::MIN_LEN_AFTER_SPLIT;

// A tree in a `BTreeMap` is a tree in the `node` module with additional invariants:
// - Keys must appear in ascending order (according to the key's type).
// - Every non-leaf node contains at least 1 element (has at least 2 children).
// - Every non-root node contains at least MIN_LEN elements.
//
// An empty map is represented either by the absence of a root node or by a
// root node that is an empty leaf.

/// An ordered map based on a [B-Tree].
///
/// B-Trees represent a fundamental compromise between cache-efficiency and actually minimizing
/// the amount of work performed in a search. In theory, a binary search tree (BST) is the optimal
/// choice for a sorted map, as a perfectly balanced BST performs the theoretical minimum amount of
/// comparisons necessary to find an element (log<sub>2</sub>n). However, in practice the way this
/// is done is *very* inefficient for modern computer architectures. In particular, every element
/// is stored in its own individually heap-allocated node. This means that every single insertion
/// triggers a heap-allocation, and every single comparison should be a cache-miss. Since these
/// are both notably expensive things to do in practice, we are forced to, at the very least,
/// reconsider the BST strategy.
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
/// to take B * log(n) comparisons, which is generally worse than a BST. In practice,
/// however, performance is excellent.
///
/// It is a logic error for a key to be modified in such a way that the key's ordering relative to
/// any other key, as determined by the [`Ord`] trait, changes while it is in the map. This is
/// normally only possible through [`Cell`], [`RefCell`], global state, I/O, or unsafe code.
/// The behavior resulting from such a logic error is not specified, but will be encapsulated to the
/// `BTreeMap` that observed the logic error and not result in undefined behavior. This could
/// include panics, incorrect results, aborts, memory leaks, and non-termination.
///
/// Iterators obtained from functions such as [`BTreeMap::iter`], [`BTreeMap::into_iter`], [`BTreeMap::values`], or
/// [`BTreeMap::keys`] produce their items in order by key, and take worst-case logarithmic and
/// amortized constant time per item returned.
///
/// [B-Tree]: https://en.wikipedia.org/wiki/B-tree
/// [`Cell`]: core::cell::Cell
/// [`RefCell`]: core::cell::RefCell
///
/// # Examples
///
/// ```
/// use std::collections::BTreeMap;
///
/// // type inference lets us omit an explicit type signature (which
/// // would be `BTreeMap<&str, &str>` in this example).
/// let mut movie_reviews = BTreeMap::new();
///
/// // review some movies.
/// movie_reviews.insert("Office Space",       "Deals with real issues in the workplace.");
/// movie_reviews.insert("Pulp Fiction",       "Masterpiece.");
/// movie_reviews.insert("The Godfather",      "Very enjoyable.");
/// movie_reviews.insert("The Blues Brothers", "Eye lyked it a lot.");
///
/// // check for a specific one.
/// if !movie_reviews.contains_key("Les Misérables") {
///     println!("We've got {} reviews, but Les Misérables ain't one.",
///              movie_reviews.len());
/// }
///
/// // oops, this review has a lot of spelling mistakes, let's delete it.
/// movie_reviews.remove("The Blues Brothers");
///
/// // look up the values associated with some keys.
/// let to_find = ["Up!", "Office Space"];
/// for movie in &to_find {
///     match movie_reviews.get(movie) {
///        Some(review) => println!("{movie}: {review}"),
///        None => println!("{movie} is unreviewed.")
///     }
/// }
///
/// // Look up the value for a key (will panic if the key is not found).
/// println!("Movie review: {}", movie_reviews["Office Space"]);
///
/// // iterate over everything.
/// for (movie, review) in &movie_reviews {
///     println!("{movie}: \"{review}\"");
/// }
/// ```
///
/// A `BTreeMap` with a known list of items can be initialized from an array:
///
/// ```
/// use std::collections::BTreeMap;
///
/// let solar_distance = BTreeMap::from([
///     ("Mercury", 0.4),
///     ("Venus", 0.7),
///     ("Earth", 1.0),
///     ("Mars", 1.5),
/// ]);
/// ```
///
/// `BTreeMap` implements an [`Entry API`], which allows for complex
/// methods of getting, setting, updating and removing keys and their values:
///
/// [`Entry API`]: BTreeMap::entry
///
/// ```
/// use std::collections::BTreeMap;
///
/// // type inference lets us omit an explicit type signature (which
/// // would be `BTreeMap<&str, u8>` in this example).
/// let mut player_stats = BTreeMap::new();
///
/// fn random_stat_buff() -> u8 {
///     // could actually return some random value here - let's just return
///     // some fixed value for now
///     42
/// }
///
/// // insert a key only if it doesn't already exist
/// player_stats.entry("health").or_insert(100);
///
/// // insert a key using a function that provides a new value only if it
/// // doesn't already exist
/// player_stats.entry("defence").or_insert_with(random_stat_buff);
///
/// // update a key, guarding against the key possibly not being set
/// let stat = player_stats.entry("attack").or_insert(100);
/// *stat += random_stat_buff();
///
/// // modify an entry before an insert with in-place mutation
/// player_stats.entry("mana").and_modify(|mana| *mana += 200).or_insert(100);
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
#[cfg_attr(not(test), rustc_diagnostic_item = "BTreeMap")]
#[rustc_insignificant_dtor]
pub struct BTreeMap<
    K,
    V,
    #[unstable(feature = "allocator_api", issue = "32838")] A: Allocator + Clone = Global,
> {
    root: Option<Root<K, V>>,
    length: usize,
    /// `ManuallyDrop` to control drop order (needs to be dropped after all the nodes).
    pub(super) alloc: ManuallyDrop<A>,
    // For dropck; the `Box` avoids making the `Unpin` impl more strict than before
    _marker: PhantomData<crate::boxed::Box<(K, V), A>>,
}

#[stable(feature = "btree_drop", since = "1.7.0")]
unsafe impl<#[may_dangle] K, #[may_dangle] V, A: Allocator + Clone> Drop for BTreeMap<K, V, A> {
    fn drop(&mut self) {
        drop(unsafe { ptr::read(self) }.into_iter())
    }
}

// FIXME: This implementation is "wrong", but changing it would be a breaking change.
// (The bounds of the automatic `UnwindSafe` implementation have been like this since Rust 1.50.)
// Maybe we can fix it nonetheless with a crater run, or if the `UnwindSafe`
// traits are deprecated, or disarmed (no longer causing hard errors) in the future.
#[stable(feature = "btree_unwindsafe", since = "1.64.0")]
impl<K, V, A: Allocator + Clone> core::panic::UnwindSafe for BTreeMap<K, V, A>
where
    A: core::panic::UnwindSafe,
    K: core::panic::RefUnwindSafe,
    V: core::panic::RefUnwindSafe,
{
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<K: Clone, V: Clone, A: Allocator + Clone> Clone for BTreeMap<K, V, A> {
    fn clone(&self) -> BTreeMap<K, V, A> {
        fn clone_subtree<'a, K: Clone, V: Clone, A: Allocator + Clone>(
            node: NodeRef<marker::Immut<'a>, K, V, marker::LeafOrInternal>,
            alloc: A,
        ) -> BTreeMap<K, V, A>
        where
            K: 'a,
            V: 'a,
        {
            match node.force() {
                Leaf(leaf) => {
                    let mut out_tree = BTreeMap {
                        root: Some(Root::new(alloc.clone())),
                        length: 0,
                        alloc: ManuallyDrop::new(alloc),
                        _marker: PhantomData,
                    };

                    {
                        let root = out_tree.root.as_mut().unwrap(); // unwrap succeeds because we just wrapped
                        let mut out_node = match root.borrow_mut().force() {
                            Leaf(leaf) => leaf,
                            Internal(_) => unreachable!(),
                        };

                        let mut in_edge = leaf.first_edge();
                        while let Ok(kv) = in_edge.right_kv() {
                            let (k, v) = kv.into_kv();
                            in_edge = kv.right_edge();

                            out_node.push(k.clone(), v.clone());
                            out_tree.length += 1;
                        }
                    }

                    out_tree
                }
                Internal(internal) => {
                    let mut out_tree =
                        clone_subtree(internal.first_edge().descend(), alloc.clone());

                    {
                        let out_root = out_tree.root.as_mut().unwrap();
                        let mut out_node = out_root.push_internal_level(alloc.clone());
                        let mut in_edge = internal.first_edge();
                        while let Ok(kv) = in_edge.right_kv() {
                            let (k, v) = kv.into_kv();
                            in_edge = kv.right_edge();

                            let k = (*k).clone();
                            let v = (*v).clone();
                            let subtree = clone_subtree(in_edge.descend(), alloc.clone());

                            // We can't destructure subtree directly
                            // because BTreeMap implements Drop
                            let (subroot, sublength) = unsafe {
                                let subtree = ManuallyDrop::new(subtree);
                                let root = ptr::read(&subtree.root);
                                let length = subtree.length;
                                (root, length)
                            };

                            out_node.push(
                                k,
                                v,
                                subroot.unwrap_or_else(|| Root::new(alloc.clone())),
                            );
                            out_tree.length += 1 + sublength;
                        }
                    }

                    out_tree
                }
            }
        }

        if self.is_empty() {
            BTreeMap::new_in((*self.alloc).clone())
        } else {
            clone_subtree(self.root.as_ref().unwrap().reborrow(), (*self.alloc).clone()) // unwrap succeeds because not empty
        }
    }
}

// Internal functionality for `BTreeSet`.
impl<K, A: Allocator + Clone> BTreeMap<K, SetValZST, A> {
    pub(super) fn replace(&mut self, key: K) -> Option<K>
    where
        K: Ord,
    {
        let (map, dormant_map) = DormantMutRef::new(self);
        let root_node =
            map.root.get_or_insert_with(|| Root::new((*map.alloc).clone())).borrow_mut();
        match root_node.search_tree::<K>(&key) {
            Found(mut kv) => Some(mem::replace(kv.key_mut(), key)),
            GoDown(handle) => {
                VacantEntry {
                    key,
                    handle: Some(handle),
                    dormant_map,
                    alloc: (*map.alloc).clone(),
                    _marker: PhantomData,
                }
                .insert(SetValZST);
                None
            }
        }
    }

    pub(super) fn get_or_insert_with<Q: ?Sized, F>(&mut self, q: &Q, f: F) -> &K
    where
        K: Borrow<Q> + Ord,
        Q: Ord,
        F: FnOnce(&Q) -> K,
    {
        let (map, dormant_map) = DormantMutRef::new(self);
        let root_node =
            map.root.get_or_insert_with(|| Root::new((*map.alloc).clone())).borrow_mut();
        match root_node.search_tree(q) {
            Found(handle) => handle.into_kv_mut().0,
            GoDown(handle) => {
                let key = f(q);
                assert!(*key.borrow() == *q, "new value is not equal");
                VacantEntry {
                    key,
                    handle: Some(handle),
                    dormant_map,
                    alloc: (*map.alloc).clone(),
                    _marker: PhantomData,
                }
                .insert_entry(SetValZST)
                .into_key()
            }
        }
    }
}

/// An iterator over the entries of a `BTreeMap`.
///
/// This `struct` is created by the [`iter`] method on [`BTreeMap`]. See its
/// documentation for more.
///
/// [`iter`]: BTreeMap::iter
#[must_use = "iterators are lazy and do nothing unless consumed"]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Iter<'a, K: 'a, V: 'a> {
    range: LazyLeafRange<marker::Immut<'a>, K, V>,
    length: usize,
}

#[stable(feature = "collection_debug", since = "1.17.0")]
impl<K: fmt::Debug, V: fmt::Debug> fmt::Debug for Iter<'_, K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.clone()).finish()
    }
}

#[stable(feature = "default_iters", since = "1.70.0")]
impl<'a, K: 'a, V: 'a> Default for Iter<'a, K, V> {
    /// Creates an empty `btree_map::Iter`.
    ///
    /// ```
    /// # use std::collections::btree_map;
    /// let iter: btree_map::Iter<'_, u8, u8> = Default::default();
    /// assert_eq!(iter.len(), 0);
    /// ```
    fn default() -> Self {
        Iter { range: Default::default(), length: 0 }
    }
}

/// A mutable iterator over the entries of a `BTreeMap`.
///
/// This `struct` is created by the [`iter_mut`] method on [`BTreeMap`]. See its
/// documentation for more.
///
/// [`iter_mut`]: BTreeMap::iter_mut
#[stable(feature = "rust1", since = "1.0.0")]
pub struct IterMut<'a, K: 'a, V: 'a> {
    range: LazyLeafRange<marker::ValMut<'a>, K, V>,
    length: usize,

    // Be invariant in `K` and `V`
    _marker: PhantomData<&'a mut (K, V)>,
}

#[must_use = "iterators are lazy and do nothing unless consumed"]
#[stable(feature = "collection_debug", since = "1.17.0")]
impl<K: fmt::Debug, V: fmt::Debug> fmt::Debug for IterMut<'_, K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let range = Iter { range: self.range.reborrow(), length: self.length };
        f.debug_list().entries(range).finish()
    }
}

#[stable(feature = "default_iters", since = "1.70.0")]
impl<'a, K: 'a, V: 'a> Default for IterMut<'a, K, V> {
    /// Creates an empty `btree_map::IterMut`.
    ///
    /// ```
    /// # use std::collections::btree_map;
    /// let iter: btree_map::IterMut<'_, u8, u8> = Default::default();
    /// assert_eq!(iter.len(), 0);
    /// ```
    fn default() -> Self {
        IterMut { range: Default::default(), length: 0, _marker: PhantomData {} }
    }
}

/// An owning iterator over the entries of a `BTreeMap`, sorted by key.
///
/// This `struct` is created by the [`into_iter`] method on [`BTreeMap`]
/// (provided by the [`IntoIterator`] trait). See its documentation for more.
///
/// [`into_iter`]: IntoIterator::into_iter
#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_insignificant_dtor]
pub struct IntoIter<
    K,
    V,
    #[unstable(feature = "allocator_api", issue = "32838")] A: Allocator + Clone = Global,
> {
    range: LazyLeafRange<marker::Dying, K, V>,
    length: usize,
    /// The BTreeMap will outlive this IntoIter so we don't care about drop order for `alloc`.
    alloc: A,
}

impl<K, V, A: Allocator + Clone> IntoIter<K, V, A> {
    /// Returns an iterator of references over the remaining items.
    #[inline]
    pub(super) fn iter(&self) -> Iter<'_, K, V> {
        Iter { range: self.range.reborrow(), length: self.length }
    }
}

#[stable(feature = "collection_debug", since = "1.17.0")]
impl<K: Debug, V: Debug, A: Allocator + Clone> Debug for IntoIter<K, V, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

#[stable(feature = "default_iters", since = "1.70.0")]
impl<K, V, A> Default for IntoIter<K, V, A>
where
    A: Allocator + Default + Clone,
{
    /// Creates an empty `btree_map::IntoIter`.
    ///
    /// ```
    /// # use std::collections::btree_map;
    /// let iter: btree_map::IntoIter<u8, u8> = Default::default();
    /// assert_eq!(iter.len(), 0);
    /// ```
    fn default() -> Self {
        IntoIter { range: Default::default(), length: 0, alloc: Default::default() }
    }
}

/// An iterator over the keys of a `BTreeMap`.
///
/// This `struct` is created by the [`keys`] method on [`BTreeMap`]. See its
/// documentation for more.
///
/// [`keys`]: BTreeMap::keys
#[must_use = "iterators are lazy and do nothing unless consumed"]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Keys<'a, K, V> {
    inner: Iter<'a, K, V>,
}

#[stable(feature = "collection_debug", since = "1.17.0")]
impl<K: fmt::Debug, V> fmt::Debug for Keys<'_, K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.clone()).finish()
    }
}

/// An iterator over the values of a `BTreeMap`.
///
/// This `struct` is created by the [`values`] method on [`BTreeMap`]. See its
/// documentation for more.
///
/// [`values`]: BTreeMap::values
#[must_use = "iterators are lazy and do nothing unless consumed"]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Values<'a, K, V> {
    inner: Iter<'a, K, V>,
}

#[stable(feature = "collection_debug", since = "1.17.0")]
impl<K, V: fmt::Debug> fmt::Debug for Values<'_, K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.clone()).finish()
    }
}

/// A mutable iterator over the values of a `BTreeMap`.
///
/// This `struct` is created by the [`values_mut`] method on [`BTreeMap`]. See its
/// documentation for more.
///
/// [`values_mut`]: BTreeMap::values_mut
#[must_use = "iterators are lazy and do nothing unless consumed"]
#[stable(feature = "map_values_mut", since = "1.10.0")]
pub struct ValuesMut<'a, K, V> {
    inner: IterMut<'a, K, V>,
}

#[stable(feature = "map_values_mut", since = "1.10.0")]
impl<K, V: fmt::Debug> fmt::Debug for ValuesMut<'_, K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.inner.iter().map(|(_, val)| val)).finish()
    }
}

/// An owning iterator over the keys of a `BTreeMap`.
///
/// This `struct` is created by the [`into_keys`] method on [`BTreeMap`].
/// See its documentation for more.
///
/// [`into_keys`]: BTreeMap::into_keys
#[must_use = "iterators are lazy and do nothing unless consumed"]
#[stable(feature = "map_into_keys_values", since = "1.54.0")]
pub struct IntoKeys<K, V, A: Allocator + Clone = Global> {
    inner: IntoIter<K, V, A>,
}

#[stable(feature = "map_into_keys_values", since = "1.54.0")]
impl<K: fmt::Debug, V, A: Allocator + Clone> fmt::Debug for IntoKeys<K, V, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.inner.iter().map(|(key, _)| key)).finish()
    }
}

/// An owning iterator over the values of a `BTreeMap`.
///
/// This `struct` is created by the [`into_values`] method on [`BTreeMap`].
/// See its documentation for more.
///
/// [`into_values`]: BTreeMap::into_values
#[must_use = "iterators are lazy and do nothing unless consumed"]
#[stable(feature = "map_into_keys_values", since = "1.54.0")]
pub struct IntoValues<
    K,
    V,
    #[unstable(feature = "allocator_api", issue = "32838")] A: Allocator + Clone = Global,
> {
    inner: IntoIter<K, V, A>,
}

#[stable(feature = "map_into_keys_values", since = "1.54.0")]
impl<K, V: fmt::Debug, A: Allocator + Clone> fmt::Debug for IntoValues<K, V, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.inner.iter().map(|(_, val)| val)).finish()
    }
}

/// An iterator over a sub-range of entries in a `BTreeMap`.
///
/// This `struct` is created by the [`range`] method on [`BTreeMap`]. See its
/// documentation for more.
///
/// [`range`]: BTreeMap::range
#[must_use = "iterators are lazy and do nothing unless consumed"]
#[stable(feature = "btree_range", since = "1.17.0")]
pub struct Range<'a, K: 'a, V: 'a> {
    inner: LeafRange<marker::Immut<'a>, K, V>,
}

#[stable(feature = "collection_debug", since = "1.17.0")]
impl<K: fmt::Debug, V: fmt::Debug> fmt::Debug for Range<'_, K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.clone()).finish()
    }
}

/// A mutable iterator over a sub-range of entries in a `BTreeMap`.
///
/// This `struct` is created by the [`range_mut`] method on [`BTreeMap`]. See its
/// documentation for more.
///
/// [`range_mut`]: BTreeMap::range_mut
#[must_use = "iterators are lazy and do nothing unless consumed"]
#[stable(feature = "btree_range", since = "1.17.0")]
pub struct RangeMut<'a, K: 'a, V: 'a> {
    inner: LeafRange<marker::ValMut<'a>, K, V>,

    // Be invariant in `K` and `V`
    _marker: PhantomData<&'a mut (K, V)>,
}

#[stable(feature = "collection_debug", since = "1.17.0")]
impl<K: fmt::Debug, V: fmt::Debug> fmt::Debug for RangeMut<'_, K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let range = Range { inner: self.inner.reborrow() };
        f.debug_list().entries(range).finish()
    }
}

impl<K, V> BTreeMap<K, V> {
    /// Makes a new, empty `BTreeMap`.
    ///
    /// Does not allocate anything on its own.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::BTreeMap;
    ///
    /// let mut map = BTreeMap::new();
    ///
    /// // entries can now be inserted into the empty map
    /// map.insert(1, "a");
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_const_stable(feature = "const_btree_new", since = "1.66.0")]
    #[inline]
    #[must_use]
    pub const fn new() -> BTreeMap<K, V> {
        BTreeMap { root: None, length: 0, alloc: ManuallyDrop::new(Global), _marker: PhantomData }
    }
}

impl<K, V, A: Allocator + Clone> BTreeMap<K, V, A> {
    /// Clears the map, removing all elements.
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
        // avoid moving the allocator
        drop(BTreeMap {
            root: mem::replace(&mut self.root, None),
            length: mem::replace(&mut self.length, 0),
            alloc: self.alloc.clone(),
            _marker: PhantomData,
        });
    }

    /// Makes a new empty BTreeMap with a reasonable choice for B.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(allocator_api)]
    /// # #![feature(btreemap_alloc)]
    /// use std::collections::BTreeMap;
    /// use std::alloc::Global;
    ///
    /// let mut map = BTreeMap::new_in(Global);
    ///
    /// // entries can now be inserted into the empty map
    /// map.insert(1, "a");
    /// ```
    #[unstable(feature = "btreemap_alloc", issue = "32838")]
    pub const fn new_in(alloc: A) -> BTreeMap<K, V, A> {
        BTreeMap { root: None, length: 0, alloc: ManuallyDrop::new(alloc), _marker: PhantomData }
    }
}

impl<K, V, A: Allocator + Clone> BTreeMap<K, V, A> {
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
    pub fn get<Q: ?Sized>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q> + Ord,
        Q: Ord,
    {
        let root_node = self.root.as_ref()?.reborrow();
        match root_node.search_tree(key) {
            Found(handle) => Some(handle.into_kv().1),
            GoDown(_) => None,
        }
    }

    /// Returns the key-value pair corresponding to the supplied key. This is
    /// potentially useful:
    /// - for key types where non-identical keys can be considered equal;
    /// - for getting the `&K` stored key value from a borrowed `&Q` lookup key; or
    /// - for getting a reference to a key with the same lifetime as the collection.
    ///
    /// The supplied key may be any borrowed form of the map's key type, but the ordering
    /// on the borrowed form *must* match the ordering on the key type.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::cmp::Ordering;
    /// use std::collections::BTreeMap;
    ///
    /// #[derive(Clone, Copy, Debug)]
    /// struct S {
    ///     id: u32,
    /// #   #[allow(unused)] // prevents a "field `name` is never read" error
    ///     name: &'static str, // ignored by equality and ordering operations
    /// }
    ///
    /// impl PartialEq for S {
    ///     fn eq(&self, other: &S) -> bool {
    ///         self.id == other.id
    ///     }
    /// }
    ///
    /// impl Eq for S {}
    ///
    /// impl PartialOrd for S {
    ///     fn partial_cmp(&self, other: &S) -> Option<Ordering> {
    ///         self.id.partial_cmp(&other.id)
    ///     }
    /// }
    ///
    /// impl Ord for S {
    ///     fn cmp(&self, other: &S) -> Ordering {
    ///         self.id.cmp(&other.id)
    ///     }
    /// }
    ///
    /// let j_a = S { id: 1, name: "Jessica" };
    /// let j_b = S { id: 1, name: "Jess" };
    /// let p = S { id: 2, name: "Paul" };
    /// assert_eq!(j_a, j_b);
    ///
    /// let mut map = BTreeMap::new();
    /// map.insert(j_a, "Paris");
    /// assert_eq!(map.get_key_value(&j_a), Some((&j_a, &"Paris")));
    /// assert_eq!(map.get_key_value(&j_b), Some((&j_a, &"Paris"))); // the notable case
    /// assert_eq!(map.get_key_value(&p), None);
    /// ```
    #[stable(feature = "map_get_key_value", since = "1.40.0")]
    pub fn get_key_value<Q: ?Sized>(&self, k: &Q) -> Option<(&K, &V)>
    where
        K: Borrow<Q> + Ord,
        Q: Ord,
    {
        let root_node = self.root.as_ref()?.reborrow();
        match root_node.search_tree(k) {
            Found(handle) => Some(handle.into_kv()),
            GoDown(_) => None,
        }
    }

    /// Returns the first key-value pair in the map.
    /// The key in this pair is the minimum key in the map.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::BTreeMap;
    ///
    /// let mut map = BTreeMap::new();
    /// assert_eq!(map.first_key_value(), None);
    /// map.insert(1, "b");
    /// map.insert(2, "a");
    /// assert_eq!(map.first_key_value(), Some((&1, &"b")));
    /// ```
    #[stable(feature = "map_first_last", since = "1.66.0")]
    pub fn first_key_value(&self) -> Option<(&K, &V)>
    where
        K: Ord,
    {
        let root_node = self.root.as_ref()?.reborrow();
        root_node.first_leaf_edge().right_kv().ok().map(Handle::into_kv)
    }

    /// Returns the first entry in the map for in-place manipulation.
    /// The key of this entry is the minimum key in the map.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::BTreeMap;
    ///
    /// let mut map = BTreeMap::new();
    /// map.insert(1, "a");
    /// map.insert(2, "b");
    /// if let Some(mut entry) = map.first_entry() {
    ///     if *entry.key() > 0 {
    ///         entry.insert("first");
    ///     }
    /// }
    /// assert_eq!(*map.get(&1).unwrap(), "first");
    /// assert_eq!(*map.get(&2).unwrap(), "b");
    /// ```
    #[stable(feature = "map_first_last", since = "1.66.0")]
    pub fn first_entry(&mut self) -> Option<OccupiedEntry<'_, K, V, A>>
    where
        K: Ord,
    {
        let (map, dormant_map) = DormantMutRef::new(self);
        let root_node = map.root.as_mut()?.borrow_mut();
        let kv = root_node.first_leaf_edge().right_kv().ok()?;
        Some(OccupiedEntry {
            handle: kv.forget_node_type(),
            dormant_map,
            alloc: (*map.alloc).clone(),
            _marker: PhantomData,
        })
    }

    /// Removes and returns the first element in the map.
    /// The key of this element is the minimum key that was in the map.
    ///
    /// # Examples
    ///
    /// Draining elements in ascending order, while keeping a usable map each iteration.
    ///
    /// ```
    /// use std::collections::BTreeMap;
    ///
    /// let mut map = BTreeMap::new();
    /// map.insert(1, "a");
    /// map.insert(2, "b");
    /// while let Some((key, _val)) = map.pop_first() {
    ///     assert!(map.iter().all(|(k, _v)| *k > key));
    /// }
    /// assert!(map.is_empty());
    /// ```
    #[stable(feature = "map_first_last", since = "1.66.0")]
    pub fn pop_first(&mut self) -> Option<(K, V)>
    where
        K: Ord,
    {
        self.first_entry().map(|entry| entry.remove_entry())
    }

    /// Returns the last key-value pair in the map.
    /// The key in this pair is the maximum key in the map.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::BTreeMap;
    ///
    /// let mut map = BTreeMap::new();
    /// map.insert(1, "b");
    /// map.insert(2, "a");
    /// assert_eq!(map.last_key_value(), Some((&2, &"a")));
    /// ```
    #[stable(feature = "map_first_last", since = "1.66.0")]
    pub fn last_key_value(&self) -> Option<(&K, &V)>
    where
        K: Ord,
    {
        let root_node = self.root.as_ref()?.reborrow();
        root_node.last_leaf_edge().left_kv().ok().map(Handle::into_kv)
    }

    /// Returns the last entry in the map for in-place manipulation.
    /// The key of this entry is the maximum key in the map.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::BTreeMap;
    ///
    /// let mut map = BTreeMap::new();
    /// map.insert(1, "a");
    /// map.insert(2, "b");
    /// if let Some(mut entry) = map.last_entry() {
    ///     if *entry.key() > 0 {
    ///         entry.insert("last");
    ///     }
    /// }
    /// assert_eq!(*map.get(&1).unwrap(), "a");
    /// assert_eq!(*map.get(&2).unwrap(), "last");
    /// ```
    #[stable(feature = "map_first_last", since = "1.66.0")]
    pub fn last_entry(&mut self) -> Option<OccupiedEntry<'_, K, V, A>>
    where
        K: Ord,
    {
        let (map, dormant_map) = DormantMutRef::new(self);
        let root_node = map.root.as_mut()?.borrow_mut();
        let kv = root_node.last_leaf_edge().left_kv().ok()?;
        Some(OccupiedEntry {
            handle: kv.forget_node_type(),
            dormant_map,
            alloc: (*map.alloc).clone(),
            _marker: PhantomData,
        })
    }

    /// Removes and returns the last element in the map.
    /// The key of this element is the maximum key that was in the map.
    ///
    /// # Examples
    ///
    /// Draining elements in descending order, while keeping a usable map each iteration.
    ///
    /// ```
    /// use std::collections::BTreeMap;
    ///
    /// let mut map = BTreeMap::new();
    /// map.insert(1, "a");
    /// map.insert(2, "b");
    /// while let Some((key, _val)) = map.pop_last() {
    ///     assert!(map.iter().all(|(k, _v)| *k < key));
    /// }
    /// assert!(map.is_empty());
    /// ```
    #[stable(feature = "map_first_last", since = "1.66.0")]
    pub fn pop_last(&mut self) -> Option<(K, V)>
    where
        K: Ord,
    {
        self.last_entry().map(|entry| entry.remove_entry())
    }

    /// Returns `true` if the map contains a value for the specified key.
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
    #[cfg_attr(not(test), rustc_diagnostic_item = "btreemap_contains_key")]
    pub fn contains_key<Q: ?Sized>(&self, key: &Q) -> bool
    where
        K: Borrow<Q> + Ord,
        Q: Ord,
    {
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
    pub fn get_mut<Q: ?Sized>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q> + Ord,
        Q: Ord,
    {
        let root_node = self.root.as_mut()?.borrow_mut();
        match root_node.search_tree(key) {
            Found(handle) => Some(handle.into_val_mut()),
            GoDown(_) => None,
        }
    }

    /// Inserts a key-value pair into the map.
    ///
    /// If the map did not have this key present, `None` is returned.
    ///
    /// If the map did have this key present, the value is updated, and the old
    /// value is returned. The key is not updated, though; this matters for
    /// types that can be `==` without being identical. See the [module-level
    /// documentation] for more.
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
    #[rustc_confusables("push", "put", "set")]
    #[cfg_attr(not(test), rustc_diagnostic_item = "btreemap_insert")]
    pub fn insert(&mut self, key: K, value: V) -> Option<V>
    where
        K: Ord,
    {
        match self.entry(key) {
            Occupied(mut entry) => Some(entry.insert(value)),
            Vacant(entry) => {
                entry.insert(value);
                None
            }
        }
    }

    /// Tries to insert a key-value pair into the map, and returns
    /// a mutable reference to the value in the entry.
    ///
    /// If the map already had this key present, nothing is updated, and
    /// an error containing the occupied entry and the value is returned.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(map_try_insert)]
    ///
    /// use std::collections::BTreeMap;
    ///
    /// let mut map = BTreeMap::new();
    /// assert_eq!(map.try_insert(37, "a").unwrap(), &"a");
    ///
    /// let err = map.try_insert(37, "b").unwrap_err();
    /// assert_eq!(err.entry.key(), &37);
    /// assert_eq!(err.entry.get(), &"a");
    /// assert_eq!(err.value, "b");
    /// ```
    #[unstable(feature = "map_try_insert", issue = "82766")]
    pub fn try_insert(&mut self, key: K, value: V) -> Result<&mut V, OccupiedError<'_, K, V, A>>
    where
        K: Ord,
    {
        match self.entry(key) {
            Occupied(entry) => Err(OccupiedError { entry, value }),
            Vacant(entry) => Ok(entry.insert(value)),
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
    #[rustc_confusables("delete", "take")]
    pub fn remove<Q: ?Sized>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q> + Ord,
        Q: Ord,
    {
        self.remove_entry(key).map(|(_, v)| v)
    }

    /// Removes a key from the map, returning the stored key and value if the key
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
    /// assert_eq!(map.remove_entry(&1), Some((1, "a")));
    /// assert_eq!(map.remove_entry(&1), None);
    /// ```
    #[stable(feature = "btreemap_remove_entry", since = "1.45.0")]
    pub fn remove_entry<Q: ?Sized>(&mut self, key: &Q) -> Option<(K, V)>
    where
        K: Borrow<Q> + Ord,
        Q: Ord,
    {
        let (map, dormant_map) = DormantMutRef::new(self);
        let root_node = map.root.as_mut()?.borrow_mut();
        match root_node.search_tree(key) {
            Found(handle) => Some(
                OccupiedEntry {
                    handle,
                    dormant_map,
                    alloc: (*map.alloc).clone(),
                    _marker: PhantomData,
                }
                .remove_entry(),
            ),
            GoDown(_) => None,
        }
    }

    /// Retains only the elements specified by the predicate.
    ///
    /// In other words, remove all pairs `(k, v)` for which `f(&k, &mut v)` returns `false`.
    /// The elements are visited in ascending key order.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::BTreeMap;
    ///
    /// let mut map: BTreeMap<i32, i32> = (0..8).map(|x| (x, x*10)).collect();
    /// // Keep only the elements with even-numbered keys.
    /// map.retain(|&k, _| k % 2 == 0);
    /// assert!(map.into_iter().eq(vec![(0, 0), (2, 20), (4, 40), (6, 60)]));
    /// ```
    #[inline]
    #[stable(feature = "btree_retain", since = "1.53.0")]
    pub fn retain<F>(&mut self, mut f: F)
    where
        K: Ord,
        F: FnMut(&K, &mut V) -> bool,
    {
        self.extract_if(|k, v| !f(k, v)).for_each(drop);
    }

    /// Moves all elements from `other` into `self`, leaving `other` empty.
    ///
    /// If a key from `other` is already present in `self`, the respective
    /// value from `self` will be overwritten with the respective value from `other`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::BTreeMap;
    ///
    /// let mut a = BTreeMap::new();
    /// a.insert(1, "a");
    /// a.insert(2, "b");
    /// a.insert(3, "c"); // Note: Key (3) also present in b.
    ///
    /// let mut b = BTreeMap::new();
    /// b.insert(3, "d"); // Note: Key (3) also present in a.
    /// b.insert(4, "e");
    /// b.insert(5, "f");
    ///
    /// a.append(&mut b);
    ///
    /// assert_eq!(a.len(), 5);
    /// assert_eq!(b.len(), 0);
    ///
    /// assert_eq!(a[&1], "a");
    /// assert_eq!(a[&2], "b");
    /// assert_eq!(a[&3], "d"); // Note: "c" has been overwritten.
    /// assert_eq!(a[&4], "e");
    /// assert_eq!(a[&5], "f");
    /// ```
    #[stable(feature = "btree_append", since = "1.11.0")]
    pub fn append(&mut self, other: &mut Self)
    where
        K: Ord,
        A: Clone,
    {
        // Do we have to append anything at all?
        if other.is_empty() {
            return;
        }

        // We can just swap `self` and `other` if `self` is empty.
        if self.is_empty() {
            mem::swap(self, other);
            return;
        }

        let self_iter = mem::replace(self, Self::new_in((*self.alloc).clone())).into_iter();
        let other_iter = mem::replace(other, Self::new_in((*self.alloc).clone())).into_iter();
        let root = self.root.get_or_insert_with(|| Root::new((*self.alloc).clone()));
        root.append_from_sorted_iters(
            self_iter,
            other_iter,
            &mut self.length,
            (*self.alloc).clone(),
        )
    }

    /// Constructs a double-ended iterator over a sub-range of elements in the map.
    /// The simplest way is to use the range syntax `min..max`, thus `range(min..max)` will
    /// yield elements from min (inclusive) to max (exclusive).
    /// The range may also be entered as `(Bound<T>, Bound<T>)`, so for example
    /// `range((Excluded(4), Included(10)))` will yield a left-exclusive, right-inclusive
    /// range from 4 to 10.
    ///
    /// # Panics
    ///
    /// Panics if range `start > end`.
    /// Panics if range `start == end` and both bounds are `Excluded`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::BTreeMap;
    /// use std::ops::Bound::Included;
    ///
    /// let mut map = BTreeMap::new();
    /// map.insert(3, "a");
    /// map.insert(5, "b");
    /// map.insert(8, "c");
    /// for (&key, &value) in map.range((Included(&4), Included(&8))) {
    ///     println!("{key}: {value}");
    /// }
    /// assert_eq!(Some((&5, &"b")), map.range(4..).next());
    /// ```
    #[stable(feature = "btree_range", since = "1.17.0")]
    pub fn range<T: ?Sized, R>(&self, range: R) -> Range<'_, K, V>
    where
        T: Ord,
        K: Borrow<T> + Ord,
        R: RangeBounds<T>,
    {
        if let Some(root) = &self.root {
            Range { inner: root.reborrow().range_search(range) }
        } else {
            Range { inner: LeafRange::none() }
        }
    }

    /// Constructs a mutable double-ended iterator over a sub-range of elements in the map.
    /// The simplest way is to use the range syntax `min..max`, thus `range(min..max)` will
    /// yield elements from min (inclusive) to max (exclusive).
    /// The range may also be entered as `(Bound<T>, Bound<T>)`, so for example
    /// `range((Excluded(4), Included(10)))` will yield a left-exclusive, right-inclusive
    /// range from 4 to 10.
    ///
    /// # Panics
    ///
    /// Panics if range `start > end`.
    /// Panics if range `start == end` and both bounds are `Excluded`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::BTreeMap;
    ///
    /// let mut map: BTreeMap<&str, i32> =
    ///     [("Alice", 0), ("Bob", 0), ("Carol", 0), ("Cheryl", 0)].into();
    /// for (_, balance) in map.range_mut("B".."Cheryl") {
    ///     *balance += 100;
    /// }
    /// for (name, balance) in &map {
    ///     println!("{name} => {balance}");
    /// }
    /// ```
    #[stable(feature = "btree_range", since = "1.17.0")]
    pub fn range_mut<T: ?Sized, R>(&mut self, range: R) -> RangeMut<'_, K, V>
    where
        T: Ord,
        K: Borrow<T> + Ord,
        R: RangeBounds<T>,
    {
        if let Some(root) = &mut self.root {
            RangeMut { inner: root.borrow_valmut().range_search(range), _marker: PhantomData }
        } else {
            RangeMut { inner: LeafRange::none(), _marker: PhantomData }
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
    /// for x in ["a", "b", "a", "c", "a", "b"] {
    ///     count.entry(x).and_modify(|curr| *curr += 1).or_insert(1);
    /// }
    ///
    /// assert_eq!(count["a"], 3);
    /// assert_eq!(count["b"], 2);
    /// assert_eq!(count["c"], 1);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn entry(&mut self, key: K) -> Entry<'_, K, V, A>
    where
        K: Ord,
    {
        let (map, dormant_map) = DormantMutRef::new(self);
        match map.root {
            None => Vacant(VacantEntry {
                key,
                handle: None,
                dormant_map,
                alloc: (*map.alloc).clone(),
                _marker: PhantomData,
            }),
            Some(ref mut root) => match root.borrow_mut().search_tree(&key) {
                Found(handle) => Occupied(OccupiedEntry {
                    handle,
                    dormant_map,
                    alloc: (*map.alloc).clone(),
                    _marker: PhantomData,
                }),
                GoDown(handle) => Vacant(VacantEntry {
                    key,
                    handle: Some(handle),
                    dormant_map,
                    alloc: (*map.alloc).clone(),
                    _marker: PhantomData,
                }),
            },
        }
    }

    /// Splits the collection into two at the given key. Returns everything after the given key,
    /// including the key.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::BTreeMap;
    ///
    /// let mut a = BTreeMap::new();
    /// a.insert(1, "a");
    /// a.insert(2, "b");
    /// a.insert(3, "c");
    /// a.insert(17, "d");
    /// a.insert(41, "e");
    ///
    /// let b = a.split_off(&3);
    ///
    /// assert_eq!(a.len(), 2);
    /// assert_eq!(b.len(), 3);
    ///
    /// assert_eq!(a[&1], "a");
    /// assert_eq!(a[&2], "b");
    ///
    /// assert_eq!(b[&3], "c");
    /// assert_eq!(b[&17], "d");
    /// assert_eq!(b[&41], "e");
    /// ```
    #[stable(feature = "btree_split_off", since = "1.11.0")]
    pub fn split_off<Q: ?Sized + Ord>(&mut self, key: &Q) -> Self
    where
        K: Borrow<Q> + Ord,
        A: Clone,
    {
        if self.is_empty() {
            return Self::new_in((*self.alloc).clone());
        }

        let total_num = self.len();
        let left_root = self.root.as_mut().unwrap(); // unwrap succeeds because not empty

        let right_root = left_root.split_off(key, (*self.alloc).clone());

        let (new_left_len, right_len) = Root::calc_split_length(total_num, &left_root, &right_root);
        self.length = new_left_len;

        BTreeMap {
            root: Some(right_root),
            length: right_len,
            alloc: self.alloc.clone(),
            _marker: PhantomData,
        }
    }

    /// Creates an iterator that visits all elements (key-value pairs) in
    /// ascending key order and uses a closure to determine if an element
    /// should be removed.
    ///
    /// If the closure returns `true`, the element is removed from the map and
    /// yielded. If the closure returns `false`, or panics, the element remains
    /// in the map and will not be yielded.
    ///
    /// The iterator also lets you mutate the value of each element in the
    /// closure, regardless of whether you choose to keep or remove it.
    ///
    /// If the returned `ExtractIf` is not exhausted, e.g. because it is dropped without iterating
    /// or the iteration short-circuits, then the remaining elements will be retained.
    /// Use [`retain`] with a negated predicate if you do not need the returned iterator.
    ///
    /// [`retain`]: BTreeMap::retain
    ///
    /// # Examples
    ///
    /// Splitting a map into even and odd keys, reusing the original map:
    ///
    /// ```
    /// #![feature(btree_extract_if)]
    /// use std::collections::BTreeMap;
    ///
    /// let mut map: BTreeMap<i32, i32> = (0..8).map(|x| (x, x)).collect();
    /// let evens: BTreeMap<_, _> = map.extract_if(|k, _v| k % 2 == 0).collect();
    /// let odds = map;
    /// assert_eq!(evens.keys().copied().collect::<Vec<_>>(), [0, 2, 4, 6]);
    /// assert_eq!(odds.keys().copied().collect::<Vec<_>>(), [1, 3, 5, 7]);
    /// ```
    #[unstable(feature = "btree_extract_if", issue = "70530")]
    pub fn extract_if<F>(&mut self, pred: F) -> ExtractIf<'_, K, V, F, A>
    where
        K: Ord,
        F: FnMut(&K, &mut V) -> bool,
    {
        let (inner, alloc) = self.extract_if_inner();
        ExtractIf { pred, inner, alloc }
    }

    pub(super) fn extract_if_inner(&mut self) -> (ExtractIfInner<'_, K, V>, A)
    where
        K: Ord,
    {
        if let Some(root) = self.root.as_mut() {
            let (root, dormant_root) = DormantMutRef::new(root);
            let front = root.borrow_mut().first_leaf_edge();
            (
                ExtractIfInner {
                    length: &mut self.length,
                    dormant_root: Some(dormant_root),
                    cur_leaf_edge: Some(front),
                },
                (*self.alloc).clone(),
            )
        } else {
            (
                ExtractIfInner {
                    length: &mut self.length,
                    dormant_root: None,
                    cur_leaf_edge: None,
                },
                (*self.alloc).clone(),
            )
        }
    }

    /// Creates a consuming iterator visiting all the keys, in sorted order.
    /// The map cannot be used after calling this.
    /// The iterator element type is `K`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::BTreeMap;
    ///
    /// let mut a = BTreeMap::new();
    /// a.insert(2, "b");
    /// a.insert(1, "a");
    ///
    /// let keys: Vec<i32> = a.into_keys().collect();
    /// assert_eq!(keys, [1, 2]);
    /// ```
    #[inline]
    #[stable(feature = "map_into_keys_values", since = "1.54.0")]
    pub fn into_keys(self) -> IntoKeys<K, V, A> {
        IntoKeys { inner: self.into_iter() }
    }

    /// Creates a consuming iterator visiting all the values, in order by key.
    /// The map cannot be used after calling this.
    /// The iterator element type is `V`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::BTreeMap;
    ///
    /// let mut a = BTreeMap::new();
    /// a.insert(1, "hello");
    /// a.insert(2, "goodbye");
    ///
    /// let values: Vec<&str> = a.into_values().collect();
    /// assert_eq!(values, ["hello", "goodbye"]);
    /// ```
    #[inline]
    #[stable(feature = "map_into_keys_values", since = "1.54.0")]
    pub fn into_values(self) -> IntoValues<K, V, A> {
        IntoValues { inner: self.into_iter() }
    }

    /// Makes a `BTreeMap` from a sorted iterator.
    pub(crate) fn bulk_build_from_sorted_iter<I>(iter: I, alloc: A) -> Self
    where
        K: Ord,
        I: IntoIterator<Item = (K, V)>,
    {
        let mut root = Root::new(alloc.clone());
        let mut length = 0;
        root.bulk_push(DedupSortedIter::new(iter.into_iter()), &mut length, alloc.clone());
        BTreeMap { root: Some(root), length, alloc: ManuallyDrop::new(alloc), _marker: PhantomData }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, K, V, A: Allocator + Clone> IntoIterator for &'a BTreeMap<K, V, A> {
    type Item = (&'a K, &'a V);
    type IntoIter = Iter<'a, K, V>;

    fn into_iter(self) -> Iter<'a, K, V> {
        self.iter()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, K: 'a, V: 'a> Iterator for Iter<'a, K, V> {
    type Item = (&'a K, &'a V);

    fn next(&mut self) -> Option<(&'a K, &'a V)> {
        if self.length == 0 {
            None
        } else {
            self.length -= 1;
            Some(unsafe { self.range.next_unchecked() })
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.length, Some(self.length))
    }

    fn last(mut self) -> Option<(&'a K, &'a V)> {
        self.next_back()
    }

    fn min(mut self) -> Option<(&'a K, &'a V)>
    where
        (&'a K, &'a V): Ord,
    {
        self.next()
    }

    fn max(mut self) -> Option<(&'a K, &'a V)>
    where
        (&'a K, &'a V): Ord,
    {
        self.next_back()
    }
}

#[stable(feature = "fused", since = "1.26.0")]
impl<K, V> FusedIterator for Iter<'_, K, V> {}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, K: 'a, V: 'a> DoubleEndedIterator for Iter<'a, K, V> {
    fn next_back(&mut self) -> Option<(&'a K, &'a V)> {
        if self.length == 0 {
            None
        } else {
            self.length -= 1;
            Some(unsafe { self.range.next_back_unchecked() })
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<K, V> ExactSizeIterator for Iter<'_, K, V> {
    fn len(&self) -> usize {
        self.length
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<K, V> Clone for Iter<'_, K, V> {
    fn clone(&self) -> Self {
        Iter { range: self.range.clone(), length: self.length }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, K, V, A: Allocator + Clone> IntoIterator for &'a mut BTreeMap<K, V, A> {
    type Item = (&'a K, &'a mut V);
    type IntoIter = IterMut<'a, K, V>;

    fn into_iter(self) -> IterMut<'a, K, V> {
        self.iter_mut()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, K, V> Iterator for IterMut<'a, K, V> {
    type Item = (&'a K, &'a mut V);

    fn next(&mut self) -> Option<(&'a K, &'a mut V)> {
        if self.length == 0 {
            None
        } else {
            self.length -= 1;
            Some(unsafe { self.range.next_unchecked() })
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.length, Some(self.length))
    }

    fn last(mut self) -> Option<(&'a K, &'a mut V)> {
        self.next_back()
    }

    fn min(mut self) -> Option<(&'a K, &'a mut V)>
    where
        (&'a K, &'a mut V): Ord,
    {
        self.next()
    }

    fn max(mut self) -> Option<(&'a K, &'a mut V)>
    where
        (&'a K, &'a mut V): Ord,
    {
        self.next_back()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, K, V> DoubleEndedIterator for IterMut<'a, K, V> {
    fn next_back(&mut self) -> Option<(&'a K, &'a mut V)> {
        if self.length == 0 {
            None
        } else {
            self.length -= 1;
            Some(unsafe { self.range.next_back_unchecked() })
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<K, V> ExactSizeIterator for IterMut<'_, K, V> {
    fn len(&self) -> usize {
        self.length
    }
}

#[stable(feature = "fused", since = "1.26.0")]
impl<K, V> FusedIterator for IterMut<'_, K, V> {}

impl<'a, K, V> IterMut<'a, K, V> {
    /// Returns an iterator of references over the remaining items.
    #[inline]
    pub(super) fn iter(&self) -> Iter<'_, K, V> {
        Iter { range: self.range.reborrow(), length: self.length }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<K, V, A: Allocator + Clone> IntoIterator for BTreeMap<K, V, A> {
    type Item = (K, V);
    type IntoIter = IntoIter<K, V, A>;

    /// Gets an owning iterator over the entries of the map, sorted by key.
    fn into_iter(self) -> IntoIter<K, V, A> {
        let mut me = ManuallyDrop::new(self);
        if let Some(root) = me.root.take() {
            let full_range = root.into_dying().full_range();

            IntoIter {
                range: full_range,
                length: me.length,
                alloc: unsafe { ManuallyDrop::take(&mut me.alloc) },
            }
        } else {
            IntoIter {
                range: LazyLeafRange::none(),
                length: 0,
                alloc: unsafe { ManuallyDrop::take(&mut me.alloc) },
            }
        }
    }
}

#[stable(feature = "btree_drop", since = "1.7.0")]
impl<K, V, A: Allocator + Clone> Drop for IntoIter<K, V, A> {
    fn drop(&mut self) {
        struct DropGuard<'a, K, V, A: Allocator + Clone>(&'a mut IntoIter<K, V, A>);

        impl<'a, K, V, A: Allocator + Clone> Drop for DropGuard<'a, K, V, A> {
            fn drop(&mut self) {
                // Continue the same loop we perform below. This only runs when unwinding, so we
                // don't have to care about panics this time (they'll abort).
                while let Some(kv) = self.0.dying_next() {
                    // SAFETY: we consume the dying handle immediately.
                    unsafe { kv.drop_key_val() };
                }
            }
        }

        while let Some(kv) = self.dying_next() {
            let guard = DropGuard(self);
            // SAFETY: we don't touch the tree before consuming the dying handle.
            unsafe { kv.drop_key_val() };
            mem::forget(guard);
        }
    }
}

impl<K, V, A: Allocator + Clone> IntoIter<K, V, A> {
    /// Core of a `next` method returning a dying KV handle,
    /// invalidated by further calls to this function and some others.
    fn dying_next(
        &mut self,
    ) -> Option<Handle<NodeRef<marker::Dying, K, V, marker::LeafOrInternal>, marker::KV>> {
        if self.length == 0 {
            self.range.deallocating_end(self.alloc.clone());
            None
        } else {
            self.length -= 1;
            Some(unsafe { self.range.deallocating_next_unchecked(self.alloc.clone()) })
        }
    }

    /// Core of a `next_back` method returning a dying KV handle,
    /// invalidated by further calls to this function and some others.
    fn dying_next_back(
        &mut self,
    ) -> Option<Handle<NodeRef<marker::Dying, K, V, marker::LeafOrInternal>, marker::KV>> {
        if self.length == 0 {
            self.range.deallocating_end(self.alloc.clone());
            None
        } else {
            self.length -= 1;
            Some(unsafe { self.range.deallocating_next_back_unchecked(self.alloc.clone()) })
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<K, V, A: Allocator + Clone> Iterator for IntoIter<K, V, A> {
    type Item = (K, V);

    fn next(&mut self) -> Option<(K, V)> {
        // SAFETY: we consume the dying handle immediately.
        self.dying_next().map(unsafe { |kv| kv.into_key_val() })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.length, Some(self.length))
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<K, V, A: Allocator + Clone> DoubleEndedIterator for IntoIter<K, V, A> {
    fn next_back(&mut self) -> Option<(K, V)> {
        // SAFETY: we consume the dying handle immediately.
        self.dying_next_back().map(unsafe { |kv| kv.into_key_val() })
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<K, V, A: Allocator + Clone> ExactSizeIterator for IntoIter<K, V, A> {
    fn len(&self) -> usize {
        self.length
    }
}

#[stable(feature = "fused", since = "1.26.0")]
impl<K, V, A: Allocator + Clone> FusedIterator for IntoIter<K, V, A> {}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, K, V> Iterator for Keys<'a, K, V> {
    type Item = &'a K;

    fn next(&mut self) -> Option<&'a K> {
        self.inner.next().map(|(k, _)| k)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }

    fn last(mut self) -> Option<&'a K> {
        self.next_back()
    }

    fn min(mut self) -> Option<&'a K>
    where
        &'a K: Ord,
    {
        self.next()
    }

    fn max(mut self) -> Option<&'a K>
    where
        &'a K: Ord,
    {
        self.next_back()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, K, V> DoubleEndedIterator for Keys<'a, K, V> {
    fn next_back(&mut self) -> Option<&'a K> {
        self.inner.next_back().map(|(k, _)| k)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<K, V> ExactSizeIterator for Keys<'_, K, V> {
    fn len(&self) -> usize {
        self.inner.len()
    }
}

#[stable(feature = "fused", since = "1.26.0")]
impl<K, V> FusedIterator for Keys<'_, K, V> {}

#[stable(feature = "rust1", since = "1.0.0")]
impl<K, V> Clone for Keys<'_, K, V> {
    fn clone(&self) -> Self {
        Keys { inner: self.inner.clone() }
    }
}

#[stable(feature = "default_iters", since = "1.70.0")]
impl<K, V> Default for Keys<'_, K, V> {
    /// Creates an empty `btree_map::Keys`.
    ///
    /// ```
    /// # use std::collections::btree_map;
    /// let iter: btree_map::Keys<'_, u8, u8> = Default::default();
    /// assert_eq!(iter.len(), 0);
    /// ```
    fn default() -> Self {
        Keys { inner: Default::default() }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, K, V> Iterator for Values<'a, K, V> {
    type Item = &'a V;

    fn next(&mut self) -> Option<&'a V> {
        self.inner.next().map(|(_, v)| v)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }

    fn last(mut self) -> Option<&'a V> {
        self.next_back()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, K, V> DoubleEndedIterator for Values<'a, K, V> {
    fn next_back(&mut self) -> Option<&'a V> {
        self.inner.next_back().map(|(_, v)| v)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<K, V> ExactSizeIterator for Values<'_, K, V> {
    fn len(&self) -> usize {
        self.inner.len()
    }
}

#[stable(feature = "fused", since = "1.26.0")]
impl<K, V> FusedIterator for Values<'_, K, V> {}

#[stable(feature = "rust1", since = "1.0.0")]
impl<K, V> Clone for Values<'_, K, V> {
    fn clone(&self) -> Self {
        Values { inner: self.inner.clone() }
    }
}

#[stable(feature = "default_iters", since = "1.70.0")]
impl<K, V> Default for Values<'_, K, V> {
    /// Creates an empty `btree_map::Values`.
    ///
    /// ```
    /// # use std::collections::btree_map;
    /// let iter: btree_map::Values<'_, u8, u8> = Default::default();
    /// assert_eq!(iter.len(), 0);
    /// ```
    fn default() -> Self {
        Values { inner: Default::default() }
    }
}

/// An iterator produced by calling `extract_if` on BTreeMap.
#[unstable(feature = "btree_extract_if", issue = "70530")]
#[must_use = "iterators are lazy and do nothing unless consumed"]
pub struct ExtractIf<
    'a,
    K,
    V,
    F,
    #[unstable(feature = "allocator_api", issue = "32838")] A: Allocator + Clone = Global,
> {
    pred: F,
    inner: ExtractIfInner<'a, K, V>,
    /// The BTreeMap will outlive this IntoIter so we don't care about drop order for `alloc`.
    alloc: A,
}

/// Most of the implementation of ExtractIf are generic over the type
/// of the predicate, thus also serving for BTreeSet::ExtractIf.
pub(super) struct ExtractIfInner<'a, K, V> {
    /// Reference to the length field in the borrowed map, updated live.
    length: &'a mut usize,
    /// Buried reference to the root field in the borrowed map.
    /// Wrapped in `Option` to allow drop handler to `take` it.
    dormant_root: Option<DormantMutRef<'a, Root<K, V>>>,
    /// Contains a leaf edge preceding the next element to be returned, or the last leaf edge.
    /// Empty if the map has no root, if iteration went beyond the last leaf edge,
    /// or if a panic occurred in the predicate.
    cur_leaf_edge: Option<Handle<NodeRef<marker::Mut<'a>, K, V, marker::Leaf>, marker::Edge>>,
}

#[unstable(feature = "btree_extract_if", issue = "70530")]
impl<K, V, F, A> fmt::Debug for ExtractIf<'_, K, V, F, A>
where
    K: fmt::Debug,
    V: fmt::Debug,
    A: Allocator + Clone,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExtractIf").field("peek", &self.inner.peek()).finish_non_exhaustive()
    }
}

#[unstable(feature = "btree_extract_if", issue = "70530")]
impl<K, V, F, A: Allocator + Clone> Iterator for ExtractIf<'_, K, V, F, A>
where
    F: FnMut(&K, &mut V) -> bool,
{
    type Item = (K, V);

    fn next(&mut self) -> Option<(K, V)> {
        self.inner.next(&mut self.pred, self.alloc.clone())
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<'a, K, V> ExtractIfInner<'a, K, V> {
    /// Allow Debug implementations to predict the next element.
    pub(super) fn peek(&self) -> Option<(&K, &V)> {
        let edge = self.cur_leaf_edge.as_ref()?;
        edge.reborrow().next_kv().ok().map(Handle::into_kv)
    }

    /// Implementation of a typical `ExtractIf::next` method, given the predicate.
    pub(super) fn next<F, A: Allocator + Clone>(&mut self, pred: &mut F, alloc: A) -> Option<(K, V)>
    where
        F: FnMut(&K, &mut V) -> bool,
    {
        while let Ok(mut kv) = self.cur_leaf_edge.take()?.next_kv() {
            let (k, v) = kv.kv_mut();
            if pred(k, v) {
                *self.length -= 1;
                let (kv, pos) = kv.remove_kv_tracking(
                    || {
                        // SAFETY: we will touch the root in a way that will not
                        // invalidate the position returned.
                        let root = unsafe { self.dormant_root.take().unwrap().awaken() };
                        root.pop_internal_level(alloc.clone());
                        self.dormant_root = Some(DormantMutRef::new(root).1);
                    },
                    alloc.clone(),
                );
                self.cur_leaf_edge = Some(pos);
                return Some(kv);
            }
            self.cur_leaf_edge = Some(kv.next_leaf_edge());
        }
        None
    }

    /// Implementation of a typical `ExtractIf::size_hint` method.
    pub(super) fn size_hint(&self) -> (usize, Option<usize>) {
        // In most of the btree iterators, `self.length` is the number of elements
        // yet to be visited. Here, it includes elements that were visited and that
        // the predicate decided not to drain. Making this upper bound more tight
        // during iteration would require an extra field.
        (0, Some(*self.length))
    }
}

#[unstable(feature = "btree_extract_if", issue = "70530")]
impl<K, V, F> FusedIterator for ExtractIf<'_, K, V, F> where F: FnMut(&K, &mut V) -> bool {}

#[stable(feature = "btree_range", since = "1.17.0")]
impl<'a, K, V> Iterator for Range<'a, K, V> {
    type Item = (&'a K, &'a V);

    fn next(&mut self) -> Option<(&'a K, &'a V)> {
        self.inner.next_checked()
    }

    fn last(mut self) -> Option<(&'a K, &'a V)> {
        self.next_back()
    }

    fn min(mut self) -> Option<(&'a K, &'a V)>
    where
        (&'a K, &'a V): Ord,
    {
        self.next()
    }

    fn max(mut self) -> Option<(&'a K, &'a V)>
    where
        (&'a K, &'a V): Ord,
    {
        self.next_back()
    }
}

#[stable(feature = "default_iters", since = "1.70.0")]
impl<K, V> Default for Range<'_, K, V> {
    /// Creates an empty `btree_map::Range`.
    ///
    /// ```
    /// # use std::collections::btree_map;
    /// let iter: btree_map::Range<'_, u8, u8> = Default::default();
    /// assert_eq!(iter.count(), 0);
    /// ```
    fn default() -> Self {
        Range { inner: Default::default() }
    }
}

#[stable(feature = "default_iters_sequel", since = "1.82.0")]
impl<K, V> Default for RangeMut<'_, K, V> {
    /// Creates an empty `btree_map::RangeMut`.
    ///
    /// ```
    /// # use std::collections::btree_map;
    /// let iter: btree_map::RangeMut<'_, u8, u8> = Default::default();
    /// assert_eq!(iter.count(), 0);
    /// ```
    fn default() -> Self {
        RangeMut { inner: Default::default(), _marker: PhantomData }
    }
}

#[stable(feature = "map_values_mut", since = "1.10.0")]
impl<'a, K, V> Iterator for ValuesMut<'a, K, V> {
    type Item = &'a mut V;

    fn next(&mut self) -> Option<&'a mut V> {
        self.inner.next().map(|(_, v)| v)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }

    fn last(mut self) -> Option<&'a mut V> {
        self.next_back()
    }
}

#[stable(feature = "map_values_mut", since = "1.10.0")]
impl<'a, K, V> DoubleEndedIterator for ValuesMut<'a, K, V> {
    fn next_back(&mut self) -> Option<&'a mut V> {
        self.inner.next_back().map(|(_, v)| v)
    }
}

#[stable(feature = "map_values_mut", since = "1.10.0")]
impl<K, V> ExactSizeIterator for ValuesMut<'_, K, V> {
    fn len(&self) -> usize {
        self.inner.len()
    }
}

#[stable(feature = "fused", since = "1.26.0")]
impl<K, V> FusedIterator for ValuesMut<'_, K, V> {}

#[stable(feature = "default_iters_sequel", since = "1.82.0")]
impl<K, V> Default for ValuesMut<'_, K, V> {
    /// Creates an empty `btree_map::ValuesMut`.
    ///
    /// ```
    /// # use std::collections::btree_map;
    /// let iter: btree_map::ValuesMut<'_, u8, u8> = Default::default();
    /// assert_eq!(iter.count(), 0);
    /// ```
    fn default() -> Self {
        ValuesMut { inner: Default::default() }
    }
}

#[stable(feature = "map_into_keys_values", since = "1.54.0")]
impl<K, V, A: Allocator + Clone> Iterator for IntoKeys<K, V, A> {
    type Item = K;

    fn next(&mut self) -> Option<K> {
        self.inner.next().map(|(k, _)| k)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }

    fn last(mut self) -> Option<K> {
        self.next_back()
    }

    fn min(mut self) -> Option<K>
    where
        K: Ord,
    {
        self.next()
    }

    fn max(mut self) -> Option<K>
    where
        K: Ord,
    {
        self.next_back()
    }
}

#[stable(feature = "map_into_keys_values", since = "1.54.0")]
impl<K, V, A: Allocator + Clone> DoubleEndedIterator for IntoKeys<K, V, A> {
    fn next_back(&mut self) -> Option<K> {
        self.inner.next_back().map(|(k, _)| k)
    }
}

#[stable(feature = "map_into_keys_values", since = "1.54.0")]
impl<K, V, A: Allocator + Clone> ExactSizeIterator for IntoKeys<K, V, A> {
    fn len(&self) -> usize {
        self.inner.len()
    }
}

#[stable(feature = "map_into_keys_values", since = "1.54.0")]
impl<K, V, A: Allocator + Clone> FusedIterator for IntoKeys<K, V, A> {}

#[stable(feature = "default_iters", since = "1.70.0")]
impl<K, V, A> Default for IntoKeys<K, V, A>
where
    A: Allocator + Default + Clone,
{
    /// Creates an empty `btree_map::IntoKeys`.
    ///
    /// ```
    /// # use std::collections::btree_map;
    /// let iter: btree_map::IntoKeys<u8, u8> = Default::default();
    /// assert_eq!(iter.len(), 0);
    /// ```
    fn default() -> Self {
        IntoKeys { inner: Default::default() }
    }
}

#[stable(feature = "map_into_keys_values", since = "1.54.0")]
impl<K, V, A: Allocator + Clone> Iterator for IntoValues<K, V, A> {
    type Item = V;

    fn next(&mut self) -> Option<V> {
        self.inner.next().map(|(_, v)| v)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }

    fn last(mut self) -> Option<V> {
        self.next_back()
    }
}

#[stable(feature = "map_into_keys_values", since = "1.54.0")]
impl<K, V, A: Allocator + Clone> DoubleEndedIterator for IntoValues<K, V, A> {
    fn next_back(&mut self) -> Option<V> {
        self.inner.next_back().map(|(_, v)| v)
    }
}

#[stable(feature = "map_into_keys_values", since = "1.54.0")]
impl<K, V, A: Allocator + Clone> ExactSizeIterator for IntoValues<K, V, A> {
    fn len(&self) -> usize {
        self.inner.len()
    }
}

#[stable(feature = "map_into_keys_values", since = "1.54.0")]
impl<K, V, A: Allocator + Clone> FusedIterator for IntoValues<K, V, A> {}

#[stable(feature = "default_iters", since = "1.70.0")]
impl<K, V, A> Default for IntoValues<K, V, A>
where
    A: Allocator + Default + Clone,
{
    /// Creates an empty `btree_map::IntoValues`.
    ///
    /// ```
    /// # use std::collections::btree_map;
    /// let iter: btree_map::IntoValues<u8, u8> = Default::default();
    /// assert_eq!(iter.len(), 0);
    /// ```
    fn default() -> Self {
        IntoValues { inner: Default::default() }
    }
}

#[stable(feature = "btree_range", since = "1.17.0")]
impl<'a, K, V> DoubleEndedIterator for Range<'a, K, V> {
    fn next_back(&mut self) -> Option<(&'a K, &'a V)> {
        self.inner.next_back_checked()
    }
}

#[stable(feature = "fused", since = "1.26.0")]
impl<K, V> FusedIterator for Range<'_, K, V> {}

#[stable(feature = "btree_range", since = "1.17.0")]
impl<K, V> Clone for Range<'_, K, V> {
    fn clone(&self) -> Self {
        Range { inner: self.inner.clone() }
    }
}

#[stable(feature = "btree_range", since = "1.17.0")]
impl<'a, K, V> Iterator for RangeMut<'a, K, V> {
    type Item = (&'a K, &'a mut V);

    fn next(&mut self) -> Option<(&'a K, &'a mut V)> {
        self.inner.next_checked()
    }

    fn last(mut self) -> Option<(&'a K, &'a mut V)> {
        self.next_back()
    }

    fn min(mut self) -> Option<(&'a K, &'a mut V)>
    where
        (&'a K, &'a mut V): Ord,
    {
        self.next()
    }

    fn max(mut self) -> Option<(&'a K, &'a mut V)>
    where
        (&'a K, &'a mut V): Ord,
    {
        self.next_back()
    }
}

#[stable(feature = "btree_range", since = "1.17.0")]
impl<'a, K, V> DoubleEndedIterator for RangeMut<'a, K, V> {
    fn next_back(&mut self) -> Option<(&'a K, &'a mut V)> {
        self.inner.next_back_checked()
    }
}

#[stable(feature = "fused", since = "1.26.0")]
impl<K, V> FusedIterator for RangeMut<'_, K, V> {}

#[stable(feature = "rust1", since = "1.0.0")]
impl<K: Ord, V> FromIterator<(K, V)> for BTreeMap<K, V> {
    /// Constructs a `BTreeMap<K, V>` from an iterator of key-value pairs.
    ///
    /// If the iterator produces any pairs with equal keys,
    /// all but one of the corresponding values will be dropped.
    fn from_iter<T: IntoIterator<Item = (K, V)>>(iter: T) -> BTreeMap<K, V> {
        let mut inputs: Vec<_> = iter.into_iter().collect();

        if inputs.is_empty() {
            return BTreeMap::new();
        }

        // use stable sort to preserve the insertion order.
        inputs.sort_by(|a, b| a.0.cmp(&b.0));
        BTreeMap::bulk_build_from_sorted_iter(inputs, Global)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<K: Ord, V, A: Allocator + Clone> Extend<(K, V)> for BTreeMap<K, V, A> {
    #[inline]
    fn extend<T: IntoIterator<Item = (K, V)>>(&mut self, iter: T) {
        iter.into_iter().for_each(move |(k, v)| {
            self.insert(k, v);
        });
    }

    #[inline]
    fn extend_one(&mut self, (k, v): (K, V)) {
        self.insert(k, v);
    }
}

#[stable(feature = "extend_ref", since = "1.2.0")]
impl<'a, K: Ord + Copy, V: Copy, A: Allocator + Clone> Extend<(&'a K, &'a V)>
    for BTreeMap<K, V, A>
{
    fn extend<I: IntoIterator<Item = (&'a K, &'a V)>>(&mut self, iter: I) {
        self.extend(iter.into_iter().map(|(&key, &value)| (key, value)));
    }

    #[inline]
    fn extend_one(&mut self, (&k, &v): (&'a K, &'a V)) {
        self.insert(k, v);
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<K: Hash, V: Hash, A: Allocator + Clone> Hash for BTreeMap<K, V, A> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_length_prefix(self.len());
        for elt in self {
            elt.hash(state);
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<K, V> Default for BTreeMap<K, V> {
    /// Creates an empty `BTreeMap`.
    fn default() -> BTreeMap<K, V> {
        BTreeMap::new()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<K: PartialEq, V: PartialEq, A: Allocator + Clone> PartialEq for BTreeMap<K, V, A> {
    fn eq(&self, other: &BTreeMap<K, V, A>) -> bool {
        self.len() == other.len() && self.iter().zip(other).all(|(a, b)| a == b)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<K: Eq, V: Eq, A: Allocator + Clone> Eq for BTreeMap<K, V, A> {}

#[stable(feature = "rust1", since = "1.0.0")]
impl<K: PartialOrd, V: PartialOrd, A: Allocator + Clone> PartialOrd for BTreeMap<K, V, A> {
    #[inline]
    fn partial_cmp(&self, other: &BTreeMap<K, V, A>) -> Option<Ordering> {
        self.iter().partial_cmp(other.iter())
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<K: Ord, V: Ord, A: Allocator + Clone> Ord for BTreeMap<K, V, A> {
    #[inline]
    fn cmp(&self, other: &BTreeMap<K, V, A>) -> Ordering {
        self.iter().cmp(other.iter())
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<K: Debug, V: Debug, A: Allocator + Clone> Debug for BTreeMap<K, V, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map().entries(self.iter()).finish()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<K, Q: ?Sized, V, A: Allocator + Clone> Index<&Q> for BTreeMap<K, V, A>
where
    K: Borrow<Q> + Ord,
    Q: Ord,
{
    type Output = V;

    /// Returns a reference to the value corresponding to the supplied key.
    ///
    /// # Panics
    ///
    /// Panics if the key is not present in the `BTreeMap`.
    #[inline]
    fn index(&self, key: &Q) -> &V {
        self.get(key).expect("no entry found for key")
    }
}

#[stable(feature = "std_collections_from_array", since = "1.56.0")]
impl<K: Ord, V, const N: usize> From<[(K, V); N]> for BTreeMap<K, V> {
    /// Converts a `[(K, V); N]` into a `BTreeMap<K, V>`.
    ///
    /// If any entries in the array have equal keys,
    /// all but one of the corresponding values will be dropped.
    ///
    /// ```
    /// use std::collections::BTreeMap;
    ///
    /// let map1 = BTreeMap::from([(1, 2), (3, 4)]);
    /// let map2: BTreeMap<_, _> = [(1, 2), (3, 4)].into();
    /// assert_eq!(map1, map2);
    /// ```
    fn from(mut arr: [(K, V); N]) -> Self {
        if N == 0 {
            return BTreeMap::new();
        }

        // use stable sort to preserve the insertion order.
        arr.sort_by(|a, b| a.0.cmp(&b.0));
        BTreeMap::bulk_build_from_sorted_iter(arr, Global)
    }
}

impl<K, V, A: Allocator + Clone> BTreeMap<K, V, A> {
    /// Gets an iterator over the entries of the map, sorted by key.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::BTreeMap;
    ///
    /// let mut map = BTreeMap::new();
    /// map.insert(3, "c");
    /// map.insert(2, "b");
    /// map.insert(1, "a");
    ///
    /// for (key, value) in map.iter() {
    ///     println!("{key}: {value}");
    /// }
    ///
    /// let (first_key, first_value) = map.iter().next().unwrap();
    /// assert_eq!((*first_key, *first_value), (1, "a"));
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn iter(&self) -> Iter<'_, K, V> {
        if let Some(root) = &self.root {
            let full_range = root.reborrow().full_range();

            Iter { range: full_range, length: self.length }
        } else {
            Iter { range: LazyLeafRange::none(), length: 0 }
        }
    }

    /// Gets a mutable iterator over the entries of the map, sorted by key.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::BTreeMap;
    ///
    /// let mut map = BTreeMap::from([
    ///    ("a", 1),
    ///    ("b", 2),
    ///    ("c", 3),
    /// ]);
    ///
    /// // add 10 to the value if the key isn't "a"
    /// for (key, value) in map.iter_mut() {
    ///     if key != &"a" {
    ///         *value += 10;
    ///     }
    /// }
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn iter_mut(&mut self) -> IterMut<'_, K, V> {
        if let Some(root) = &mut self.root {
            let full_range = root.borrow_valmut().full_range();

            IterMut { range: full_range, length: self.length, _marker: PhantomData }
        } else {
            IterMut { range: LazyLeafRange::none(), length: 0, _marker: PhantomData }
        }
    }

    /// Gets an iterator over the keys of the map, in sorted order.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::BTreeMap;
    ///
    /// let mut a = BTreeMap::new();
    /// a.insert(2, "b");
    /// a.insert(1, "a");
    ///
    /// let keys: Vec<_> = a.keys().cloned().collect();
    /// assert_eq!(keys, [1, 2]);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn keys(&self) -> Keys<'_, K, V> {
        Keys { inner: self.iter() }
    }

    /// Gets an iterator over the values of the map, in order by key.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::BTreeMap;
    ///
    /// let mut a = BTreeMap::new();
    /// a.insert(1, "hello");
    /// a.insert(2, "goodbye");
    ///
    /// let values: Vec<&str> = a.values().cloned().collect();
    /// assert_eq!(values, ["hello", "goodbye"]);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn values(&self) -> Values<'_, K, V> {
        Values { inner: self.iter() }
    }

    /// Gets a mutable iterator over the values of the map, in order by key.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::BTreeMap;
    ///
    /// let mut a = BTreeMap::new();
    /// a.insert(1, String::from("hello"));
    /// a.insert(2, String::from("goodbye"));
    ///
    /// for value in a.values_mut() {
    ///     value.push_str("!");
    /// }
    ///
    /// let values: Vec<String> = a.values().cloned().collect();
    /// assert_eq!(values, [String::from("hello!"),
    ///                     String::from("goodbye!")]);
    /// ```
    #[stable(feature = "map_values_mut", since = "1.10.0")]
    pub fn values_mut(&mut self) -> ValuesMut<'_, K, V> {
        ValuesMut { inner: self.iter_mut() }
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
    #[must_use]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_const_unstable(
        feature = "const_btree_len",
        issue = "71835",
        implied_by = "const_btree_new"
    )]
    #[rustc_confusables("length", "size")]
    pub const fn len(&self) -> usize {
        self.length
    }

    /// Returns `true` if the map contains no elements.
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
    #[must_use]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_const_unstable(
        feature = "const_btree_len",
        issue = "71835",
        implied_by = "const_btree_new"
    )]
    pub const fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns a [`Cursor`] pointing at the gap before the smallest key
    /// greater than the given bound.
    ///
    /// Passing `Bound::Included(x)` will return a cursor pointing to the
    /// gap before the smallest key greater than or equal to `x`.
    ///
    /// Passing `Bound::Excluded(x)` will return a cursor pointing to the
    /// gap before the smallest key greater than `x`.
    ///
    /// Passing `Bound::Unbounded` will return a cursor pointing to the
    /// gap before the smallest key in the map.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(btree_cursors)]
    ///
    /// use std::collections::BTreeMap;
    /// use std::ops::Bound;
    ///
    /// let map = BTreeMap::from([
    ///     (1, "a"),
    ///     (2, "b"),
    ///     (3, "c"),
    ///     (4, "d"),
    /// ]);
    ///
    /// let cursor = map.lower_bound(Bound::Included(&2));
    /// assert_eq!(cursor.peek_prev(), Some((&1, &"a")));
    /// assert_eq!(cursor.peek_next(), Some((&2, &"b")));
    ///
    /// let cursor = map.lower_bound(Bound::Excluded(&2));
    /// assert_eq!(cursor.peek_prev(), Some((&2, &"b")));
    /// assert_eq!(cursor.peek_next(), Some((&3, &"c")));
    ///
    /// let cursor = map.lower_bound(Bound::Unbounded);
    /// assert_eq!(cursor.peek_prev(), None);
    /// assert_eq!(cursor.peek_next(), Some((&1, &"a")));
    /// ```
    #[unstable(feature = "btree_cursors", issue = "107540")]
    pub fn lower_bound<Q: ?Sized>(&self, bound: Bound<&Q>) -> Cursor<'_, K, V>
    where
        K: Borrow<Q> + Ord,
        Q: Ord,
    {
        let root_node = match self.root.as_ref() {
            None => return Cursor { current: None, root: None },
            Some(root) => root.reborrow(),
        };
        let edge = root_node.lower_bound(SearchBound::from_range(bound));
        Cursor { current: Some(edge), root: self.root.as_ref() }
    }

    /// Returns a [`CursorMut`] pointing at the gap before the smallest key
    /// greater than the given bound.
    ///
    /// Passing `Bound::Included(x)` will return a cursor pointing to the
    /// gap before the smallest key greater than or equal to `x`.
    ///
    /// Passing `Bound::Excluded(x)` will return a cursor pointing to the
    /// gap before the smallest key greater than `x`.
    ///
    /// Passing `Bound::Unbounded` will return a cursor pointing to the
    /// gap before the smallest key in the map.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(btree_cursors)]
    ///
    /// use std::collections::BTreeMap;
    /// use std::ops::Bound;
    ///
    /// let mut map = BTreeMap::from([
    ///     (1, "a"),
    ///     (2, "b"),
    ///     (3, "c"),
    ///     (4, "d"),
    /// ]);
    ///
    /// let mut cursor = map.lower_bound_mut(Bound::Included(&2));
    /// assert_eq!(cursor.peek_prev(), Some((&1, &mut "a")));
    /// assert_eq!(cursor.peek_next(), Some((&2, &mut "b")));
    ///
    /// let mut cursor = map.lower_bound_mut(Bound::Excluded(&2));
    /// assert_eq!(cursor.peek_prev(), Some((&2, &mut "b")));
    /// assert_eq!(cursor.peek_next(), Some((&3, &mut "c")));
    ///
    /// let mut cursor = map.lower_bound_mut(Bound::Unbounded);
    /// assert_eq!(cursor.peek_prev(), None);
    /// assert_eq!(cursor.peek_next(), Some((&1, &mut "a")));
    /// ```
    #[unstable(feature = "btree_cursors", issue = "107540")]
    pub fn lower_bound_mut<Q: ?Sized>(&mut self, bound: Bound<&Q>) -> CursorMut<'_, K, V, A>
    where
        K: Borrow<Q> + Ord,
        Q: Ord,
    {
        let (root, dormant_root) = DormantMutRef::new(&mut self.root);
        let root_node = match root.as_mut() {
            None => {
                return CursorMut {
                    inner: CursorMutKey {
                        current: None,
                        root: dormant_root,
                        length: &mut self.length,
                        alloc: &mut *self.alloc,
                    },
                };
            }
            Some(root) => root.borrow_mut(),
        };
        let edge = root_node.lower_bound(SearchBound::from_range(bound));
        CursorMut {
            inner: CursorMutKey {
                current: Some(edge),
                root: dormant_root,
                length: &mut self.length,
                alloc: &mut *self.alloc,
            },
        }
    }

    /// Returns a [`Cursor`] pointing at the gap after the greatest key
    /// smaller than the given bound.
    ///
    /// Passing `Bound::Included(x)` will return a cursor pointing to the
    /// gap after the greatest key smaller than or equal to `x`.
    ///
    /// Passing `Bound::Excluded(x)` will return a cursor pointing to the
    /// gap after the greatest key smaller than `x`.
    ///
    /// Passing `Bound::Unbounded` will return a cursor pointing to the
    /// gap after the greatest key in the map.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(btree_cursors)]
    ///
    /// use std::collections::BTreeMap;
    /// use std::ops::Bound;
    ///
    /// let map = BTreeMap::from([
    ///     (1, "a"),
    ///     (2, "b"),
    ///     (3, "c"),
    ///     (4, "d"),
    /// ]);
    ///
    /// let cursor = map.upper_bound(Bound::Included(&3));
    /// assert_eq!(cursor.peek_prev(), Some((&3, &"c")));
    /// assert_eq!(cursor.peek_next(), Some((&4, &"d")));
    ///
    /// let cursor = map.upper_bound(Bound::Excluded(&3));
    /// assert_eq!(cursor.peek_prev(), Some((&2, &"b")));
    /// assert_eq!(cursor.peek_next(), Some((&3, &"c")));
    ///
    /// let cursor = map.upper_bound(Bound::Unbounded);
    /// assert_eq!(cursor.peek_prev(), Some((&4, &"d")));
    /// assert_eq!(cursor.peek_next(), None);
    /// ```
    #[unstable(feature = "btree_cursors", issue = "107540")]
    pub fn upper_bound<Q: ?Sized>(&self, bound: Bound<&Q>) -> Cursor<'_, K, V>
    where
        K: Borrow<Q> + Ord,
        Q: Ord,
    {
        let root_node = match self.root.as_ref() {
            None => return Cursor { current: None, root: None },
            Some(root) => root.reborrow(),
        };
        let edge = root_node.upper_bound(SearchBound::from_range(bound));
        Cursor { current: Some(edge), root: self.root.as_ref() }
    }

    /// Returns a [`CursorMut`] pointing at the gap after the greatest key
    /// smaller than the given bound.
    ///
    /// Passing `Bound::Included(x)` will return a cursor pointing to the
    /// gap after the greatest key smaller than or equal to `x`.
    ///
    /// Passing `Bound::Excluded(x)` will return a cursor pointing to the
    /// gap after the greatest key smaller than `x`.
    ///
    /// Passing `Bound::Unbounded` will return a cursor pointing to the
    /// gap after the greatest key in the map.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(btree_cursors)]
    ///
    /// use std::collections::BTreeMap;
    /// use std::ops::Bound;
    ///
    /// let mut map = BTreeMap::from([
    ///     (1, "a"),
    ///     (2, "b"),
    ///     (3, "c"),
    ///     (4, "d"),
    /// ]);
    ///
    /// let mut cursor = map.upper_bound_mut(Bound::Included(&3));
    /// assert_eq!(cursor.peek_prev(), Some((&3, &mut "c")));
    /// assert_eq!(cursor.peek_next(), Some((&4, &mut "d")));
    ///
    /// let mut cursor = map.upper_bound_mut(Bound::Excluded(&3));
    /// assert_eq!(cursor.peek_prev(), Some((&2, &mut "b")));
    /// assert_eq!(cursor.peek_next(), Some((&3, &mut "c")));
    ///
    /// let mut cursor = map.upper_bound_mut(Bound::Unbounded);
    /// assert_eq!(cursor.peek_prev(), Some((&4, &mut "d")));
    /// assert_eq!(cursor.peek_next(), None);
    /// ```
    #[unstable(feature = "btree_cursors", issue = "107540")]
    pub fn upper_bound_mut<Q: ?Sized>(&mut self, bound: Bound<&Q>) -> CursorMut<'_, K, V, A>
    where
        K: Borrow<Q> + Ord,
        Q: Ord,
    {
        let (root, dormant_root) = DormantMutRef::new(&mut self.root);
        let root_node = match root.as_mut() {
            None => {
                return CursorMut {
                    inner: CursorMutKey {
                        current: None,
                        root: dormant_root,
                        length: &mut self.length,
                        alloc: &mut *self.alloc,
                    },
                };
            }
            Some(root) => root.borrow_mut(),
        };
        let edge = root_node.upper_bound(SearchBound::from_range(bound));
        CursorMut {
            inner: CursorMutKey {
                current: Some(edge),
                root: dormant_root,
                length: &mut self.length,
                alloc: &mut *self.alloc,
            },
        }
    }
}

/// A cursor over a `BTreeMap`.
///
/// A `Cursor` is like an iterator, except that it can freely seek back-and-forth.
///
/// Cursors always point to a gap between two elements in the map, and can
/// operate on the two immediately adjacent elements.
///
/// A `Cursor` is created with the [`BTreeMap::lower_bound`] and [`BTreeMap::upper_bound`] methods.
#[unstable(feature = "btree_cursors", issue = "107540")]
pub struct Cursor<'a, K: 'a, V: 'a> {
    // If current is None then it means the tree has not been allocated yet.
    current: Option<Handle<NodeRef<marker::Immut<'a>, K, V, marker::Leaf>, marker::Edge>>,
    root: Option<&'a node::Root<K, V>>,
}

#[unstable(feature = "btree_cursors", issue = "107540")]
impl<K, V> Clone for Cursor<'_, K, V> {
    fn clone(&self) -> Self {
        let Cursor { current, root } = *self;
        Cursor { current, root }
    }
}

#[unstable(feature = "btree_cursors", issue = "107540")]
impl<K: Debug, V: Debug> Debug for Cursor<'_, K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Cursor")
    }
}

/// A cursor over a `BTreeMap` with editing operations.
///
/// A `Cursor` is like an iterator, except that it can freely seek back-and-forth, and can
/// safely mutate the map during iteration. This is because the lifetime of its yielded
/// references is tied to its own lifetime, instead of just the underlying map. This means
/// cursors cannot yield multiple elements at once.
///
/// Cursors always point to a gap between two elements in the map, and can
/// operate on the two immediately adjacent elements.
///
/// A `CursorMut` is created with the [`BTreeMap::lower_bound_mut`] and [`BTreeMap::upper_bound_mut`]
/// methods.
#[unstable(feature = "btree_cursors", issue = "107540")]
pub struct CursorMut<
    'a,
    K: 'a,
    V: 'a,
    #[unstable(feature = "allocator_api", issue = "32838")] A = Global,
> {
    inner: CursorMutKey<'a, K, V, A>,
}

#[unstable(feature = "btree_cursors", issue = "107540")]
impl<K: Debug, V: Debug, A> Debug for CursorMut<'_, K, V, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("CursorMut")
    }
}

/// A cursor over a `BTreeMap` with editing operations, and which allows
/// mutating the key of elements.
///
/// A `Cursor` is like an iterator, except that it can freely seek back-and-forth, and can
/// safely mutate the map during iteration. This is because the lifetime of its yielded
/// references is tied to its own lifetime, instead of just the underlying map. This means
/// cursors cannot yield multiple elements at once.
///
/// Cursors always point to a gap between two elements in the map, and can
/// operate on the two immediately adjacent elements.
///
/// A `CursorMutKey` is created from a [`CursorMut`] with the
/// [`CursorMut::with_mutable_key`] method.
///
/// # Safety
///
/// Since this cursor allows mutating keys, you must ensure that the `BTreeMap`
/// invariants are maintained. Specifically:
///
/// * The key of the newly inserted element must be unique in the tree.
/// * All keys in the tree must remain in sorted order.
#[unstable(feature = "btree_cursors", issue = "107540")]
pub struct CursorMutKey<
    'a,
    K: 'a,
    V: 'a,
    #[unstable(feature = "allocator_api", issue = "32838")] A = Global,
> {
    // If current is None then it means the tree has not been allocated yet.
    current: Option<Handle<NodeRef<marker::Mut<'a>, K, V, marker::Leaf>, marker::Edge>>,
    root: DormantMutRef<'a, Option<node::Root<K, V>>>,
    length: &'a mut usize,
    alloc: &'a mut A,
}

#[unstable(feature = "btree_cursors", issue = "107540")]
impl<K: Debug, V: Debug, A> Debug for CursorMutKey<'_, K, V, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("CursorMutKey")
    }
}

impl<'a, K, V> Cursor<'a, K, V> {
    /// Advances the cursor to the next gap, returning the key and value of the
    /// element that it moved over.
    ///
    /// If the cursor is already at the end of the map then `None` is returned
    /// and the cursor is not moved.
    #[unstable(feature = "btree_cursors", issue = "107540")]
    pub fn next(&mut self) -> Option<(&'a K, &'a V)> {
        let current = self.current.take()?;
        match current.next_kv() {
            Ok(kv) => {
                let result = kv.into_kv();
                self.current = Some(kv.next_leaf_edge());
                Some(result)
            }
            Err(root) => {
                self.current = Some(root.last_leaf_edge());
                None
            }
        }
    }

    /// Advances the cursor to the previous gap, returning the key and value of
    /// the element that it moved over.
    ///
    /// If the cursor is already at the start of the map then `None` is returned
    /// and the cursor is not moved.
    #[unstable(feature = "btree_cursors", issue = "107540")]
    pub fn prev(&mut self) -> Option<(&'a K, &'a V)> {
        let current = self.current.take()?;
        match current.next_back_kv() {
            Ok(kv) => {
                let result = kv.into_kv();
                self.current = Some(kv.next_back_leaf_edge());
                Some(result)
            }
            Err(root) => {
                self.current = Some(root.first_leaf_edge());
                None
            }
        }
    }

    /// Returns a reference to the key and value of the next element without
    /// moving the cursor.
    ///
    /// If the cursor is at the end of the map then `None` is returned.
    #[unstable(feature = "btree_cursors", issue = "107540")]
    pub fn peek_next(&self) -> Option<(&'a K, &'a V)> {
        self.clone().next()
    }

    /// Returns a reference to the key and value of the previous element
    /// without moving the cursor.
    ///
    /// If the cursor is at the start of the map then `None` is returned.
    #[unstable(feature = "btree_cursors", issue = "107540")]
    pub fn peek_prev(&self) -> Option<(&'a K, &'a V)> {
        self.clone().prev()
    }
}

impl<'a, K, V, A> CursorMut<'a, K, V, A> {
    /// Advances the cursor to the next gap, returning the key and value of the
    /// element that it moved over.
    ///
    /// If the cursor is already at the end of the map then `None` is returned
    /// and the cursor is not moved.
    #[unstable(feature = "btree_cursors", issue = "107540")]
    pub fn next(&mut self) -> Option<(&K, &mut V)> {
        let (k, v) = self.inner.next()?;
        Some((&*k, v))
    }

    /// Advances the cursor to the previous gap, returning the key and value of
    /// the element that it moved over.
    ///
    /// If the cursor is already at the start of the map then `None` is returned
    /// and the cursor is not moved.
    #[unstable(feature = "btree_cursors", issue = "107540")]
    pub fn prev(&mut self) -> Option<(&K, &mut V)> {
        let (k, v) = self.inner.prev()?;
        Some((&*k, v))
    }

    /// Returns a reference to the key and value of the next element without
    /// moving the cursor.
    ///
    /// If the cursor is at the end of the map then `None` is returned.
    #[unstable(feature = "btree_cursors", issue = "107540")]
    pub fn peek_next(&mut self) -> Option<(&K, &mut V)> {
        let (k, v) = self.inner.peek_next()?;
        Some((&*k, v))
    }

    /// Returns a reference to the key and value of the previous element
    /// without moving the cursor.
    ///
    /// If the cursor is at the start of the map then `None` is returned.
    #[unstable(feature = "btree_cursors", issue = "107540")]
    pub fn peek_prev(&mut self) -> Option<(&K, &mut V)> {
        let (k, v) = self.inner.peek_prev()?;
        Some((&*k, v))
    }

    /// Returns a read-only cursor pointing to the same location as the
    /// `CursorMut`.
    ///
    /// The lifetime of the returned `Cursor` is bound to that of the
    /// `CursorMut`, which means it cannot outlive the `CursorMut` and that the
    /// `CursorMut` is frozen for the lifetime of the `Cursor`.
    #[unstable(feature = "btree_cursors", issue = "107540")]
    pub fn as_cursor(&self) -> Cursor<'_, K, V> {
        self.inner.as_cursor()
    }

    /// Converts the cursor into a [`CursorMutKey`], which allows mutating
    /// the key of elements in the tree.
    ///
    /// # Safety
    ///
    /// Since this cursor allows mutating keys, you must ensure that the `BTreeMap`
    /// invariants are maintained. Specifically:
    ///
    /// * The key of the newly inserted element must be unique in the tree.
    /// * All keys in the tree must remain in sorted order.
    #[unstable(feature = "btree_cursors", issue = "107540")]
    pub unsafe fn with_mutable_key(self) -> CursorMutKey<'a, K, V, A> {
        self.inner
    }
}

impl<'a, K, V, A> CursorMutKey<'a, K, V, A> {
    /// Advances the cursor to the next gap, returning the key and value of the
    /// element that it moved over.
    ///
    /// If the cursor is already at the end of the map then `None` is returned
    /// and the cursor is not moved.
    #[unstable(feature = "btree_cursors", issue = "107540")]
    pub fn next(&mut self) -> Option<(&mut K, &mut V)> {
        let current = self.current.take()?;
        match current.next_kv() {
            Ok(mut kv) => {
                // SAFETY: The key/value pointers remain valid even after the
                // cursor is moved forward. The lifetimes then prevent any
                // further access to the cursor.
                let (k, v) = unsafe { kv.reborrow_mut().into_kv_mut() };
                let (k, v) = (k as *mut _, v as *mut _);
                self.current = Some(kv.next_leaf_edge());
                Some(unsafe { (&mut *k, &mut *v) })
            }
            Err(root) => {
                self.current = Some(root.last_leaf_edge());
                None
            }
        }
    }

    /// Advances the cursor to the previous gap, returning the key and value of
    /// the element that it moved over.
    ///
    /// If the cursor is already at the start of the map then `None` is returned
    /// and the cursor is not moved.
    #[unstable(feature = "btree_cursors", issue = "107540")]
    pub fn prev(&mut self) -> Option<(&mut K, &mut V)> {
        let current = self.current.take()?;
        match current.next_back_kv() {
            Ok(mut kv) => {
                // SAFETY: The key/value pointers remain valid even after the
                // cursor is moved forward. The lifetimes then prevent any
                // further access to the cursor.
                let (k, v) = unsafe { kv.reborrow_mut().into_kv_mut() };
                let (k, v) = (k as *mut _, v as *mut _);
                self.current = Some(kv.next_back_leaf_edge());
                Some(unsafe { (&mut *k, &mut *v) })
            }
            Err(root) => {
                self.current = Some(root.first_leaf_edge());
                None
            }
        }
    }

    /// Returns a reference to the key and value of the next element without
    /// moving the cursor.
    ///
    /// If the cursor is at the end of the map then `None` is returned.
    #[unstable(feature = "btree_cursors", issue = "107540")]
    pub fn peek_next(&mut self) -> Option<(&mut K, &mut V)> {
        let current = self.current.as_mut()?;
        // SAFETY: We're not using this to mutate the tree.
        let kv = unsafe { current.reborrow_mut() }.next_kv().ok()?.into_kv_mut();
        Some(kv)
    }

    /// Returns a reference to the key and value of the previous element
    /// without moving the cursor.
    ///
    /// If the cursor is at the start of the map then `None` is returned.
    #[unstable(feature = "btree_cursors", issue = "107540")]
    pub fn peek_prev(&mut self) -> Option<(&mut K, &mut V)> {
        let current = self.current.as_mut()?;
        // SAFETY: We're not using this to mutate the tree.
        let kv = unsafe { current.reborrow_mut() }.next_back_kv().ok()?.into_kv_mut();
        Some(kv)
    }

    /// Returns a read-only cursor pointing to the same location as the
    /// `CursorMutKey`.
    ///
    /// The lifetime of the returned `Cursor` is bound to that of the
    /// `CursorMutKey`, which means it cannot outlive the `CursorMutKey` and that the
    /// `CursorMutKey` is frozen for the lifetime of the `Cursor`.
    #[unstable(feature = "btree_cursors", issue = "107540")]
    pub fn as_cursor(&self) -> Cursor<'_, K, V> {
        Cursor {
            // SAFETY: The tree is immutable while the cursor exists.
            root: unsafe { self.root.reborrow_shared().as_ref() },
            current: self.current.as_ref().map(|current| current.reborrow()),
        }
    }
}

// Now the tree editing operations
impl<'a, K: Ord, V, A: Allocator + Clone> CursorMutKey<'a, K, V, A> {
    /// Inserts a new key-value pair into the map in the gap that the
    /// cursor is currently pointing to.
    ///
    /// After the insertion the cursor will be pointing at the gap before the
    /// newly inserted element.
    ///
    /// # Safety
    ///
    /// You must ensure that the `BTreeMap` invariants are maintained.
    /// Specifically:
    ///
    /// * The key of the newly inserted element must be unique in the tree.
    /// * All keys in the tree must remain in sorted order.
    #[unstable(feature = "btree_cursors", issue = "107540")]
    pub unsafe fn insert_after_unchecked(&mut self, key: K, value: V) {
        let edge = match self.current.take() {
            None => {
                // Tree is empty, allocate a new root.
                // SAFETY: We have no other reference to the tree.
                let root = unsafe { self.root.reborrow() };
                debug_assert!(root.is_none());
                let mut node = NodeRef::new_leaf(self.alloc.clone());
                // SAFETY: We don't touch the root while the handle is alive.
                let handle = unsafe { node.borrow_mut().push_with_handle(key, value) };
                *root = Some(node.forget_type());
                *self.length += 1;
                self.current = Some(handle.left_edge());
                return;
            }
            Some(current) => current,
        };

        let handle = edge.insert_recursing(key, value, self.alloc.clone(), |ins| {
            drop(ins.left);
            // SAFETY: The handle to the newly inserted value is always on a
            // leaf node, so adding a new root node doesn't invalidate it.
            let root = unsafe { self.root.reborrow().as_mut().unwrap() };
            root.push_internal_level(self.alloc.clone()).push(ins.kv.0, ins.kv.1, ins.right)
        });
        self.current = Some(handle.left_edge());
        *self.length += 1;
    }

    /// Inserts a new key-value pair into the map in the gap that the
    /// cursor is currently pointing to.
    ///
    /// After the insertion the cursor will be pointing at the gap after the
    /// newly inserted element.
    ///
    /// # Safety
    ///
    /// You must ensure that the `BTreeMap` invariants are maintained.
    /// Specifically:
    ///
    /// * The key of the newly inserted element must be unique in the tree.
    /// * All keys in the tree must remain in sorted order.
    #[unstable(feature = "btree_cursors", issue = "107540")]
    pub unsafe fn insert_before_unchecked(&mut self, key: K, value: V) {
        let edge = match self.current.take() {
            None => {
                // SAFETY: We have no other reference to the tree.
                match unsafe { self.root.reborrow() } {
                    root @ None => {
                        // Tree is empty, allocate a new root.
                        let mut node = NodeRef::new_leaf(self.alloc.clone());
                        // SAFETY: We don't touch the root while the handle is alive.
                        let handle = unsafe { node.borrow_mut().push_with_handle(key, value) };
                        *root = Some(node.forget_type());
                        *self.length += 1;
                        self.current = Some(handle.right_edge());
                        return;
                    }
                    Some(root) => root.borrow_mut().last_leaf_edge(),
                }
            }
            Some(current) => current,
        };

        let handle = edge.insert_recursing(key, value, self.alloc.clone(), |ins| {
            drop(ins.left);
            // SAFETY: The handle to the newly inserted value is always on a
            // leaf node, so adding a new root node doesn't invalidate it.
            let root = unsafe { self.root.reborrow().as_mut().unwrap() };
            root.push_internal_level(self.alloc.clone()).push(ins.kv.0, ins.kv.1, ins.right)
        });
        self.current = Some(handle.right_edge());
        *self.length += 1;
    }

    /// Inserts a new key-value pair into the map in the gap that the
    /// cursor is currently pointing to.
    ///
    /// After the insertion the cursor will be pointing at the gap before the
    /// newly inserted element.
    ///
    /// If the inserted key is not greater than the key before the cursor
    /// (if any), or if it not less than the key after the cursor (if any),
    /// then an [`UnorderedKeyError`] is returned since this would
    /// invalidate the [`Ord`] invariant between the keys of the map.
    #[unstable(feature = "btree_cursors", issue = "107540")]
    pub fn insert_after(&mut self, key: K, value: V) -> Result<(), UnorderedKeyError> {
        if let Some((prev, _)) = self.peek_prev() {
            if &key <= prev {
                return Err(UnorderedKeyError {});
            }
        }
        if let Some((next, _)) = self.peek_next() {
            if &key >= next {
                return Err(UnorderedKeyError {});
            }
        }
        unsafe {
            self.insert_after_unchecked(key, value);
        }
        Ok(())
    }

    /// Inserts a new key-value pair into the map in the gap that the
    /// cursor is currently pointing to.
    ///
    /// After the insertion the cursor will be pointing at the gap after the
    /// newly inserted element.
    ///
    /// If the inserted key is not greater than the key before the cursor
    /// (if any), or if it not less than the key after the cursor (if any),
    /// then an [`UnorderedKeyError`] is returned since this would
    /// invalidate the [`Ord`] invariant between the keys of the map.
    #[unstable(feature = "btree_cursors", issue = "107540")]
    pub fn insert_before(&mut self, key: K, value: V) -> Result<(), UnorderedKeyError> {
        if let Some((prev, _)) = self.peek_prev() {
            if &key <= prev {
                return Err(UnorderedKeyError {});
            }
        }
        if let Some((next, _)) = self.peek_next() {
            if &key >= next {
                return Err(UnorderedKeyError {});
            }
        }
        unsafe {
            self.insert_before_unchecked(key, value);
        }
        Ok(())
    }

    /// Removes the next element from the `BTreeMap`.
    ///
    /// The element that was removed is returned. The cursor position is
    /// unchanged (before the removed element).
    #[unstable(feature = "btree_cursors", issue = "107540")]
    pub fn remove_next(&mut self) -> Option<(K, V)> {
        let current = self.current.take()?;
        if current.reborrow().next_kv().is_err() {
            self.current = Some(current);
            return None;
        }
        let mut emptied_internal_root = false;
        let (kv, pos) = current
            .next_kv()
            // This should be unwrap(), but that doesn't work because NodeRef
            // doesn't implement Debug. The condition is checked above.
            .ok()?
            .remove_kv_tracking(|| emptied_internal_root = true, self.alloc.clone());
        self.current = Some(pos);
        *self.length -= 1;
        if emptied_internal_root {
            // SAFETY: This is safe since current does not point within the now
            // empty root node.
            let root = unsafe { self.root.reborrow().as_mut().unwrap() };
            root.pop_internal_level(self.alloc.clone());
        }
        Some(kv)
    }

    /// Removes the preceding element from the `BTreeMap`.
    ///
    /// The element that was removed is returned. The cursor position is
    /// unchanged (after the removed element).
    #[unstable(feature = "btree_cursors", issue = "107540")]
    pub fn remove_prev(&mut self) -> Option<(K, V)> {
        let current = self.current.take()?;
        if current.reborrow().next_back_kv().is_err() {
            self.current = Some(current);
            return None;
        }
        let mut emptied_internal_root = false;
        let (kv, pos) = current
            .next_back_kv()
            // This should be unwrap(), but that doesn't work because NodeRef
            // doesn't implement Debug. The condition is checked above.
            .ok()?
            .remove_kv_tracking(|| emptied_internal_root = true, self.alloc.clone());
        self.current = Some(pos);
        *self.length -= 1;
        if emptied_internal_root {
            // SAFETY: This is safe since current does not point within the now
            // empty root node.
            let root = unsafe { self.root.reborrow().as_mut().unwrap() };
            root.pop_internal_level(self.alloc.clone());
        }
        Some(kv)
    }
}

impl<'a, K: Ord, V, A: Allocator + Clone> CursorMut<'a, K, V, A> {
    /// Inserts a new key-value pair into the map in the gap that the
    /// cursor is currently pointing to.
    ///
    /// After the insertion the cursor will be pointing at the gap after the
    /// newly inserted element.
    ///
    /// # Safety
    ///
    /// You must ensure that the `BTreeMap` invariants are maintained.
    /// Specifically:
    ///
    /// * The key of the newly inserted element must be unique in the tree.
    /// * All keys in the tree must remain in sorted order.
    #[unstable(feature = "btree_cursors", issue = "107540")]
    pub unsafe fn insert_after_unchecked(&mut self, key: K, value: V) {
        unsafe { self.inner.insert_after_unchecked(key, value) }
    }

    /// Inserts a new key-value pair into the map in the gap that the
    /// cursor is currently pointing to.
    ///
    /// After the insertion the cursor will be pointing at the gap after the
    /// newly inserted element.
    ///
    /// # Safety
    ///
    /// You must ensure that the `BTreeMap` invariants are maintained.
    /// Specifically:
    ///
    /// * The key of the newly inserted element must be unique in the tree.
    /// * All keys in the tree must remain in sorted order.
    #[unstable(feature = "btree_cursors", issue = "107540")]
    pub unsafe fn insert_before_unchecked(&mut self, key: K, value: V) {
        unsafe { self.inner.insert_before_unchecked(key, value) }
    }

    /// Inserts a new key-value pair into the map in the gap that the
    /// cursor is currently pointing to.
    ///
    /// After the insertion the cursor will be pointing at the gap before the
    /// newly inserted element.
    ///
    /// If the inserted key is not greater than the key before the cursor
    /// (if any), or if it not less than the key after the cursor (if any),
    /// then an [`UnorderedKeyError`] is returned since this would
    /// invalidate the [`Ord`] invariant between the keys of the map.
    #[unstable(feature = "btree_cursors", issue = "107540")]
    pub fn insert_after(&mut self, key: K, value: V) -> Result<(), UnorderedKeyError> {
        self.inner.insert_after(key, value)
    }

    /// Inserts a new key-value pair into the map in the gap that the
    /// cursor is currently pointing to.
    ///
    /// After the insertion the cursor will be pointing at the gap after the
    /// newly inserted element.
    ///
    /// If the inserted key is not greater than the key before the cursor
    /// (if any), or if it not less than the key after the cursor (if any),
    /// then an [`UnorderedKeyError`] is returned since this would
    /// invalidate the [`Ord`] invariant between the keys of the map.
    #[unstable(feature = "btree_cursors", issue = "107540")]
    pub fn insert_before(&mut self, key: K, value: V) -> Result<(), UnorderedKeyError> {
        self.inner.insert_before(key, value)
    }

    /// Removes the next element from the `BTreeMap`.
    ///
    /// The element that was removed is returned. The cursor position is
    /// unchanged (before the removed element).
    #[unstable(feature = "btree_cursors", issue = "107540")]
    pub fn remove_next(&mut self) -> Option<(K, V)> {
        self.inner.remove_next()
    }

    /// Removes the preceding element from the `BTreeMap`.
    ///
    /// The element that was removed is returned. The cursor position is
    /// unchanged (after the removed element).
    #[unstable(feature = "btree_cursors", issue = "107540")]
    pub fn remove_prev(&mut self) -> Option<(K, V)> {
        self.inner.remove_prev()
    }
}

/// Error type returned by [`CursorMut::insert_before`] and
/// [`CursorMut::insert_after`] if the key being inserted is not properly
/// ordered with regards to adjacent keys.
#[derive(Clone, PartialEq, Eq, Debug)]
#[unstable(feature = "btree_cursors", issue = "107540")]
pub struct UnorderedKeyError {}

#[unstable(feature = "btree_cursors", issue = "107540")]
impl fmt::Display for UnorderedKeyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "key is not properly ordered relative to neighbors")
    }
}

#[unstable(feature = "btree_cursors", issue = "107540")]
impl Error for UnorderedKeyError {}

#[cfg(test)]
mod tests;
