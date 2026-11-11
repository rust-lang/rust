use std::cmp::Ordering;
use std::mem;

use rustc_index::{Idx, IndexVec};

#[cfg(test)]
mod tests;

/// Simple implementation of a union-find data structure, i.e. a disjoint-set
/// forest.
#[derive(Debug)]
pub struct UnionFind<Key: Idx> {
    table: IndexVec<Key, UnionFindEntry<Key>>,
}

#[derive(Debug)]
struct UnionFindEntry<Key> {
    /// Transitively points towards the "root" of the set containing this key.
    ///
    /// Invariant: A root key is its own parent.
    parent: Key,
    /// When merging two "root" keys, their ranks determine which key becomes
    /// the new root, to prevent the parent tree from becoming unnecessarily
    /// tall. See [`UnionFind::unify`] for details.
    rank: u32,
}

impl<Key: Idx> UnionFind<Key> {
    /// Creates a new disjoint-set forest containing the keys `0..num_keys`.
    /// Initially, every key is part of its own one-element set.
    pub fn new(num_keys: usize) -> Self {
        // Initially, every key is the root of its own set, so its parent is itself.
        Self { table: IndexVec::from_fn_n(|key| UnionFindEntry { parent: key, rank: 0 }, num_keys) }
    }

    /// Returns the "root" key of the disjoint-set containing the given key.
    /// If two keys have the same root, they belong to the same set.
    ///
    /// Also updates internal data structures to make subsequent `find`
    /// operations faster.
    pub fn find(&mut self, key: Key) -> Key {
        // Loop until we find a key that is its own parent.
        let mut curr = key;
        while let parent = self.table[curr].parent
            && curr != parent
        {
            // Perform "path compression" by peeking one layer ahead, and
            // setting the current key's parent to that value.
            // (This works even when `parent` is the root of its set, because
            // of the invariant that a root is its own parent.)
            let parent_parent = self.table[parent].parent;
            self.table[curr].parent = parent_parent;

            // Advance by one step and continue.
            curr = parent;
        }
        curr
    }

    /// Merges the set containing `a` and the set containing `b` into one set.
    ///
    /// Returns the common root of both keys, after the merge.
    pub fn unify(&mut self, a: Key, b: Key) -> Key {
        let mut a = self.find(a);
        let mut b = self.find(b);

        // If both keys have the same root, they're already in the same set,
        // so there's nothing more to do.
        if a == b {
            return a;
        };

        // Ensure that `a` has strictly greater rank, swapping if necessary.
        // If both keys have the same rank, increment the rank of `a` so that
        // future unifications will also prefer `a`, leading to flatter trees.
        match Ord::cmp(&self.table[a].rank, &self.table[b].rank) {
            Ordering::Less => mem::swap(&mut a, &mut b),
            Ordering::Equal => self.table[a].rank += 1,
            Ordering::Greater => {}
        }

        debug_assert!(self.table[a].rank > self.table[b].rank);
        debug_assert_eq!(self.table[b].parent, b);

        // Make `a` the parent of `b`.
        self.table[b].parent = a;

        a
    }

    /// Takes a "snapshot" of the current state of this disjoint-set forest, in
    /// the form of a vector that directly maps each key to its current root.
    pub fn snapshot(&mut self) -> IndexVec<Key, Key> {
        self.table.indices().map(|key| self.find(key)).collect()
    }
}
