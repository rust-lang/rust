use core::borrow::Borrow;
use core::cmp::Ordering;

use super::node::{marker, ForceResult::*, Handle, NodeRef};

use SearchResult::*;

pub enum SearchResult<BorrowType, K, V, FoundType, GoDownType> {
    Found(Handle<NodeRef<BorrowType, K, V, FoundType>, marker::KV>),
    GoDown(Handle<NodeRef<BorrowType, K, V, GoDownType>, marker::Edge>),
}

pub enum IndexResult {
    KV(usize),
    Edge(usize),
}

impl<BorrowType, K, V> NodeRef<BorrowType, K, V, marker::LeafOrInternal> {
    /// Looks up a given key in a (sub)tree headed by the node, recursively.
    /// Returns a `Found` with the handle of the matching KV, if any. Otherwise,
    /// returns a `GoDown` with the handle of the leaf edge where the key belongs.
    ///
    /// The result is meaningful only if the tree is ordered by key, like the tree
    /// in a `BTreeMap` is.
    pub fn search_tree<Q: ?Sized>(
        mut self,
        key: &Q,
    ) -> SearchResult<BorrowType, K, V, marker::LeafOrInternal, marker::Leaf>
    where
        Q: Ord,
        K: Borrow<Q>,
    {
        loop {
            self = match self.search_node(key) {
                Found(handle) => return Found(handle),
                GoDown(handle) => match handle.force() {
                    Leaf(leaf) => return GoDown(leaf),
                    Internal(internal) => internal.descend(),
                },
            }
        }
    }
}

impl<BorrowType, K, V, Type> NodeRef<BorrowType, K, V, Type> {
    /// Looks up a given key in the node, without recursion.
    /// Returns a `Found` with the handle of the matching KV, if any. Otherwise,
    /// returns a `GoDown` with the handle of the edge where the key might be found
    /// (if the node is internal) or where the key can be inserted.
    ///
    /// The result is meaningful only if the tree is ordered by key, like the tree
    /// in a `BTreeMap` is.
    pub fn search_node<Q: ?Sized>(self, key: &Q) -> SearchResult<BorrowType, K, V, Type, Type>
    where
        Q: Ord,
        K: Borrow<Q>,
    {
        match self.find_index(key) {
            IndexResult::KV(idx) => Found(unsafe { Handle::new_kv(self, idx) }),
            IndexResult::Edge(idx) => GoDown(unsafe { Handle::new_edge(self, idx) }),
        }
    }

    /// Returns either the KV index in the node at which the key (or an equivalent)
    /// exists, or the edge index where the key belongs.
    ///
    /// The result is meaningful only if the tree is ordered by key, like the tree
    /// in a `BTreeMap` is.
    fn find_index<Q: ?Sized>(&self, key: &Q) -> IndexResult
    where
        Q: Ord,
        K: Borrow<Q>,
    {
        // This function is defined over all borrow types (immutable, mutable, owned).
        // Using `keys_at()` is fine here even if BorrowType is mutable, as all we return
        // is an index -- not a reference.
        let len = self.len();
        for i in 0..len {
            let k = unsafe { self.reborrow().key_at(i) };
            match key.cmp(k.borrow()) {
                Ordering::Greater => {}
                Ordering::Equal => return IndexResult::KV(i),
                Ordering::Less => return IndexResult::Edge(i),
            }
        }
        IndexResult::Edge(len)
    }
}
