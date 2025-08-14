//! This module implements the `UniMap`, which is a way to get efficient mappings
//! optimized for the setting of `tree_borrows/tree.rs`.
//!
//! A `UniKeyMap<K>` is a (slow) mapping from `K` to `UniIndex`,
//! and `UniValMap<V>` is a (fast) mapping from `UniIndex` to `V`.
//! Thus a pair `(UniKeyMap<K>, UniValMap<V>)` acts as a virtual `HashMap<K, V>`.
//!
//! Because of the asymmetry in access time, the use-case for `UniMap` is the following:
//! a tuple `(UniKeyMap<K>, Vec<UniValMap<V>>)` is much more efficient than
//! the equivalent `Vec<HashMap<K, V>>` it represents if all maps have similar
//! sets of keys.

#![allow(dead_code)]

use std::hash::Hash;
use std::mem;

use rustc_data_structures::fx::FxHashMap;

use crate::helpers::ToUsize;

/// Intermediate key between a UniKeyMap and a UniValMap.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UniIndex {
    idx: u32,
}

/// From K to UniIndex
#[derive(Debug, Clone, Default)]
pub struct UniKeyMap<K> {
    /// Underlying map that does all the hard work.
    /// Key invariant: the contents of `deassigned` are disjoint from the
    /// keys of `mapping`, and together they form the set of contiguous integers
    /// `0 .. (mapping.len() + deassigned.len())`.
    mapping: FxHashMap<K, u32>,
    /// Indexes that can be reused: memory gain when the map gets sparse
    /// due to many deletions.
    deassigned: Vec<u32>,
}

/// From UniIndex to V
#[derive(Debug, Clone, Eq)]
pub struct UniValMap<V> {
    /// The mapping data. Thanks to Vec we get both fast accesses, and
    /// a memory-optimal representation if there are few deletions.
    data: Vec<Option<V>>,
}

impl<V: PartialEq> UniValMap<V> {
    /// Exact equality of two maps.
    /// Less accurate but faster than `equivalent`, mostly because
    /// of the fast path when the lengths are different.
    pub fn identical(&self, other: &Self) -> bool {
        self.data == other.data
    }

    /// Equality up to trailing `None`s of two maps, i.e.
    /// do they represent the same mapping ?
    pub fn equivalent(&self, other: &Self) -> bool {
        let min_len = self.data.len().min(other.data.len());
        self.data[min_len..].iter().all(Option::is_none)
            && other.data[min_len..].iter().all(Option::is_none)
            && (self.data[..min_len] == other.data[..min_len])
    }
}

impl<V: PartialEq> PartialEq for UniValMap<V> {
    /// 2023-05: We found that using `equivalent` rather than `identical`
    /// in the equality testing of the `RangeMap` is neutral for most
    /// benchmarks, while being quite beneficial for `zip-equal`
    /// and to a lesser extent for `unicode`, `slice-get-unchecked` and
    /// `backtraces` as well.
    fn eq(&self, other: &Self) -> bool {
        self.equivalent(other)
    }
}

impl<V> Default for UniValMap<V> {
    fn default() -> Self {
        Self { data: Vec::default() }
    }
}

impl<K> UniKeyMap<K>
where
    K: Hash + Eq,
{
    /// How many keys/index pairs are currently active.
    pub fn len(&self) -> usize {
        self.mapping.len()
    }

    /// Whether this key has an associated index or not.
    pub fn contains_key(&self, key: &K) -> bool {
        self.mapping.contains_key(key)
    }

    /// Assign this key to a new index. Panics if the key is already assigned,
    /// use `get_or_insert` for a version that instead returns the existing
    /// assignment.
    #[track_caller]
    pub fn insert(&mut self, key: K) -> UniIndex {
        // We want an unused index. First we attempt to find one from `deassigned`,
        // and if `deassigned` is empty we generate a fresh index.
        let idx = self.deassigned.pop().unwrap_or_else(|| {
            // `deassigned` is empty, so all keys in use are already in `mapping`.
            // The next available key is `mapping.len()`.
            self.mapping.len().try_into().expect("UniMap ran out of useable keys")
        });
        if self.mapping.insert(key, idx).is_some() {
            panic!(
                "This key is already assigned to a different index; either use `get_or_insert` instead if you care about this data, or first call `remove` to undo the preexisting assignment."
            );
        };
        UniIndex { idx }
    }

    /// If it exists, the index this key maps to.
    pub fn get(&self, key: &K) -> Option<UniIndex> {
        self.mapping.get(key).map(|&idx| UniIndex { idx })
    }

    /// Either get a previously existing entry, or create a new one if it
    /// is not yet present.
    pub fn get_or_insert(&mut self, key: K) -> UniIndex {
        self.get(&key).unwrap_or_else(|| self.insert(key))
    }

    /// Return whatever index this key was using to the deassigned pool.
    ///
    /// Note: calling this function can be dangerous. If the index still exists
    /// somewhere in a `UniValMap` and is reassigned by the `UniKeyMap` then
    /// it will inherit the old value of a completely unrelated key.
    /// If you `UniKeyMap::remove` a key you should make sure to also `UniValMap::remove`
    /// the associated `UniIndex` from ALL `UniValMap`s.
    ///
    /// Example of such behavior:
    /// ```rust,ignore (private type can't be doctested)
    /// let mut keymap = UniKeyMap::<char>::default();
    /// let mut valmap = UniValMap::<char>::default();
    /// // Insert 'a' -> _ -> 'A'
    /// let idx_a = keymap.insert('a');
    /// valmap.insert(idx_a, 'A');
    /// // Remove 'a' -> _, but forget to remove _ -> 'A'
    /// keymap.remove(&'a');
    /// // valmap.remove(idx_a); // If we uncomment this line the issue is fixed
    /// // Insert 'b' -> _
    /// let idx_b = keymap.insert('b');
    /// let val_b = valmap.get(idx_b);
    /// assert_eq!(val_b, Some('A')); // Oh no
    /// // assert_eq!(val_b, None); // This is what we would have expected
    /// ```
    pub fn remove(&mut self, key: &K) {
        if let Some(idx) = self.mapping.remove(key) {
            self.deassigned.push(idx);
        }
    }
}

impl<V> UniValMap<V> {
    /// Whether this index has an associated value.
    pub fn contains_idx(&self, idx: UniIndex) -> bool {
        self.data.get(idx.idx.to_usize()).and_then(Option::as_ref).is_some()
    }

    /// Reserve enough space to insert the value at the right index.
    fn extend_to_length(&mut self, len: usize) {
        if len > self.data.len() {
            let nb = len - self.data.len();
            self.data.reserve(nb);
            for _ in 0..nb {
                self.data.push(None);
            }
        }
    }

    /// Assign a value to the index. Permanently overwrites any previous value.
    pub fn insert(&mut self, idx: UniIndex, val: V) {
        self.extend_to_length(idx.idx.to_usize() + 1);
        self.data[idx.idx.to_usize()] = Some(val)
    }

    /// Get the value at this index, if it exists.
    pub fn get(&self, idx: UniIndex) -> Option<&V> {
        self.data.get(idx.idx.to_usize()).and_then(Option::as_ref)
    }

    /// Get the value at this index mutably, if it exists.
    pub fn get_mut(&mut self, idx: UniIndex) -> Option<&mut V> {
        self.data.get_mut(idx.idx.to_usize()).and_then(Option::as_mut)
    }

    /// Delete any value associated with this index.
    /// Returns None if the value was not present, otherwise
    /// returns the previously stored value.
    pub fn remove(&mut self, idx: UniIndex) -> Option<V> {
        if idx.idx.to_usize() >= self.data.len() {
            return None;
        }
        let mut res = None;
        mem::swap(&mut res, &mut self.data[idx.idx.to_usize()]);
        res
    }
}

/// An access to a single value of the map.
pub struct UniEntry<'a, V> {
    inner: &'a mut Option<V>,
}

impl<'a, V> UniValMap<V> {
    /// Get a wrapper around a mutable access to the value corresponding to `idx`.
    pub fn entry(&'a mut self, idx: UniIndex) -> UniEntry<'a, V> {
        self.extend_to_length(idx.idx.to_usize() + 1);
        UniEntry { inner: &mut self.data[idx.idx.to_usize()] }
    }
}

impl<'a, V> UniEntry<'a, V> {
    /// Insert in the map and get the value.
    pub fn or_insert(&mut self, default: V) -> &mut V {
        if self.inner.is_none() {
            *self.inner = Some(default);
        }
        self.inner.as_mut().unwrap()
    }

    pub fn get(&self) -> Option<&V> {
        self.inner.as_ref()
    }
}

mod tests {
    use super::*;

    #[test]
    fn extend_to_length() {
        let mut km = UniValMap::<char>::default();
        km.extend_to_length(10);
        assert!(km.data.len() == 10);
        km.extend_to_length(0);
        assert!(km.data.len() == 10);
        km.extend_to_length(10);
        assert!(km.data.len() == 10);
        km.extend_to_length(11);
        assert!(km.data.len() == 11);
    }

    #[derive(Default)]
    struct MapWitness<K, V> {
        key: UniKeyMap<K>,
        val: UniValMap<V>,
        map: FxHashMap<K, V>,
    }

    impl<K, V> MapWitness<K, V>
    where
        K: Copy + Hash + Eq,
        V: Copy + Eq + std::fmt::Debug,
    {
        fn insert(&mut self, k: K, v: V) {
            // UniMap
            let i = self.key.get_or_insert(k);
            self.val.insert(i, v);
            // HashMap
            self.map.insert(k, v);
            // Consistency: nothing to check
        }

        fn get(&self, k: &K) {
            // UniMap
            let v1 = self.key.get(k).and_then(|i| self.val.get(i));
            // HashMap
            let v2 = self.map.get(k);
            // Consistency
            assert_eq!(v1, v2);
        }

        fn get_mut(&mut self, k: &K) {
            // UniMap
            let v1 = self.key.get(k).and_then(|i| self.val.get_mut(i));
            // HashMap
            let v2 = self.map.get_mut(k);
            // Consistency
            assert_eq!(v1, v2);
        }
        fn remove(&mut self, k: &K) {
            // UniMap
            if let Some(i) = self.key.get(k) {
                self.val.remove(i);
            }
            self.key.remove(k);
            // HashMap
            self.map.remove(k);
            // Consistency: nothing to check
        }
    }

    #[test]
    fn consistency_small() {
        let mut m = MapWitness::<u64, char>::default();
        m.insert(1, 'a');
        m.insert(2, 'b');
        m.get(&1);
        m.get_mut(&2);
        m.remove(&2);
        m.insert(1, 'c');
        m.get(&1);
        m.insert(3, 'd');
        m.insert(4, 'e');
        m.insert(4, 'f');
        m.get(&2);
        m.get(&3);
        m.get(&4);
        m.get(&5);
        m.remove(&100);
        m.get_mut(&100);
        m.get(&100);
    }

    #[test]
    fn consistency_large() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        let mut map = MapWitness::<u64, u64>::default();
        for i in 0..1000 {
            i.hash(&mut hasher);
            let rng = hasher.finish();
            let op = rng.is_multiple_of(3);
            let key = (rng / 2) % 50;
            let val = (rng / 100) % 1000;
            if op {
                map.insert(key, val);
            } else {
                map.get(&key);
            }
        }
    }
}
