//! A variant of `SortedMap` that preserves insertion order.

use std::borrow::Borrow;
use std::hash::{Hash, Hasher};
use std::iter::FromIterator;

use crate::stable_hasher::{HashStable, StableHasher};
use rustc_index::vec::{Idx, IndexVec};

/// An indexed multi-map that preserves insertion order while permitting both *O*(log *n*) lookup of
/// an item by key and *O*(1) lookup by index.
///
/// This data structure is a hybrid of an [`IndexVec`] and a [`SortedMap`]. Like `IndexVec`,
/// `SortedIndexMultiMap` assigns a typed index to each item while preserving insertion order.
/// Like `SortedMap`, `SortedIndexMultiMap` has efficient lookup of items by key. However, this
/// is accomplished by sorting an array of item indices instead of the items themselves.
///
/// Unlike `SortedMap`, this data structure can hold multiple equivalent items at once, so the
/// `get_by_key` method and its variants return an iterator instead of an `Option`. Equivalent
/// items will be yielded in insertion order.
///
/// Unlike a general-purpose map like `BTreeSet` or `HashSet`, `SortedMap` and
/// `SortedIndexMultiMap` require *O*(*n*) time to insert a single item. This is because we may need
/// to insert into the middle of the sorted array. Users should avoid mutating this data structure
/// in-place.
///
/// [`SortedMap`]: super::SortedMap
#[derive(Clone, Debug)]
pub struct SortedIndexMultiMap<I: Idx, K, V> {
    /// The elements of the map in insertion order.
    items: IndexVec<I, (K, V)>,

    /// Indices of the items in the set, sorted by the item's key.
    idx_sorted_by_item_key: Vec<I>,
}

impl<I: Idx, K: Ord, V> SortedIndexMultiMap<I, K, V> {
    pub fn new() -> Self {
        SortedIndexMultiMap { items: IndexVec::new(), idx_sorted_by_item_key: Vec::new() }
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Returns an iterator over the items in the map in insertion order.
    pub fn into_iter(self) -> impl DoubleEndedIterator<Item = (K, V)> {
        self.items.into_iter()
    }

    /// Returns an iterator over the items in the map in insertion order along with their indices.
    pub fn into_iter_enumerated(self) -> impl DoubleEndedIterator<Item = (I, (K, V))> {
        self.items.into_iter_enumerated()
    }

    /// Returns an iterator over the items in the map in insertion order.
    pub fn iter(&self) -> impl '_ + DoubleEndedIterator<Item = (&K, &V)> {
        self.items.iter().map(|(ref k, ref v)| (k, v))
    }

    /// Returns an iterator over the items in the map in insertion order along with their indices.
    pub fn iter_enumerated(&self) -> impl '_ + DoubleEndedIterator<Item = (I, (&K, &V))> {
        self.items.iter_enumerated().map(|(i, (ref k, ref v))| (i, (k, v)))
    }

    /// Returns the item in the map with the given index.
    pub fn get(&self, idx: I) -> Option<&(K, V)> {
        self.items.get(idx)
    }

    /// Returns an iterator over the items in the map that are equal to `key`.
    ///
    /// If there are multiple items that are equivalent to `key`, they will be yielded in
    /// insertion order.
    pub fn get_by_key<Q: 'a>(&'a self, key: &Q) -> impl 'a + Iterator<Item = &'a V>
    where
        Q: Ord + ?Sized,
        K: Borrow<Q>,
    {
        self.get_by_key_enumerated(key).map(|(_, v)| v)
    }

    /// Returns an iterator over the items in the map that are equal to `key` along with their
    /// indices.
    ///
    /// If there are multiple items that are equivalent to `key`, they will be yielded in
    /// insertion order.
    pub fn get_by_key_enumerated<Q>(&self, key: &Q) -> impl '_ + Iterator<Item = (I, &V)>
    where
        Q: Ord + ?Sized,
        K: Borrow<Q>,
    {
        // FIXME: This should be in the standard library as `equal_range`. See rust-lang/rfcs#2184.
        match self.binary_search_idx(key) {
            Err(_) => self.idxs_to_items_enumerated(&[]),

            Ok(idx) => {
                let start = self.find_lower_bound(key, idx);
                let end = self.find_upper_bound(key, idx);
                self.idxs_to_items_enumerated(&self.idx_sorted_by_item_key[start..end])
            }
        }
    }

    fn binary_search_idx<Q>(&self, key: &Q) -> Result<usize, usize>
    where
        Q: Ord + ?Sized,
        K: Borrow<Q>,
    {
        self.idx_sorted_by_item_key.binary_search_by(|&idx| self.items[idx].0.borrow().cmp(key))
    }

    /// Returns the index into the `idx_sorted_by_item_key` array of the first item equal to
    /// `key`.
    ///
    /// `initial` must be an index into that same array for an item that is equal to `key`.
    fn find_lower_bound<Q>(&self, key: &Q, initial: usize) -> usize
    where
        Q: Ord + ?Sized,
        K: Borrow<Q>,
    {
        debug_assert!(self.items[self.idx_sorted_by_item_key[initial]].0.borrow() == key);

        // FIXME: At present, this uses linear search, meaning lookup is only `O(log n)` if duplicate
        // entries are rare. It would be better to start with a linear search for the common case but
        // fall back to an exponential search if many duplicates are found. This applies to
        // `upper_bound` as well.
        let mut start = initial;
        while start != 0 && self.items[self.idx_sorted_by_item_key[start - 1]].0.borrow() == key {
            start -= 1;
        }

        start
    }

    /// Returns the index into the `idx_sorted_by_item_key` array of the first item greater than
    /// `key`, or `self.len()` if no such item exists.
    ///
    /// `initial` must be an index into that same array for an item that is equal to `key`.
    fn find_upper_bound<Q>(&self, key: &Q, initial: usize) -> usize
    where
        Q: Ord + ?Sized,
        K: Borrow<Q>,
    {
        debug_assert!(self.items[self.idx_sorted_by_item_key[initial]].0.borrow() == key);

        // See the FIXME for `find_lower_bound`.
        let mut end = initial + 1;
        let len = self.items.len();
        while end < len && self.items[self.idx_sorted_by_item_key[end]].0.borrow() == key {
            end += 1;
        }

        end
    }

    fn idxs_to_items_enumerated(&'a self, idxs: &'a [I]) -> impl 'a + Iterator<Item = (I, &'a V)> {
        idxs.iter().map(move |&idx| (idx, &self.items[idx].1))
    }
}

impl<I: Idx, K: Eq, V: Eq> Eq for SortedIndexMultiMap<I, K, V> {}
impl<I: Idx, K: PartialEq, V: PartialEq> PartialEq for SortedIndexMultiMap<I, K, V> {
    fn eq(&self, other: &Self) -> bool {
        // No need to compare the sorted index. If the items are the same, the index will be too.
        self.items == other.items
    }
}

impl<I: Idx, K, V> Hash for SortedIndexMultiMap<I, K, V>
where
    K: Hash,
    V: Hash,
{
    fn hash<H: Hasher>(&self, hasher: &mut H) {
        self.items.hash(hasher)
    }
}
impl<I: Idx, K, V, C> HashStable<C> for SortedIndexMultiMap<I, K, V>
where
    K: HashStable<C>,
    V: HashStable<C>,
{
    fn hash_stable(&self, ctx: &mut C, hasher: &mut StableHasher) {
        self.items.hash_stable(ctx, hasher)
    }
}

impl<I: Idx, K: Ord, V> FromIterator<(K, V)> for SortedIndexMultiMap<I, K, V> {
    fn from_iter<J>(iter: J) -> Self
    where
        J: IntoIterator<Item = (K, V)>,
    {
        let items = IndexVec::from_iter(iter);
        let mut idx_sorted_by_item_key: Vec<_> = items.indices().collect();

        // `sort_by_key` is stable, so insertion order is preserved for duplicate items.
        idx_sorted_by_item_key.sort_by_key(|&idx| &items[idx].0);

        SortedIndexMultiMap { items, idx_sorted_by_item_key }
    }
}

impl<I: Idx, K, V> std::ops::Index<I> for SortedIndexMultiMap<I, K, V> {
    type Output = V;

    fn index(&self, idx: I) -> &Self::Output {
        &self.items[idx].1
    }
}

#[cfg(tests)]
mod tests;
