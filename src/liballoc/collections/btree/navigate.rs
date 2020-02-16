use core::ptr;

use super::node::{marker, ForceResult::*, Handle, NodeRef};
use super::unwrap_unchecked;

macro_rules! def_next {
    { unsafe fn $name:ident : $next_kv:ident $next_edge:ident $initial_leaf_edge:ident } => {
        /// Given a leaf edge handle into an immutable tree, returns a handle to the next
        /// leaf edge and references to the key and value between these edges.
        /// Unsafe because the caller must ensure that the given leaf edge has a successor.
        unsafe fn $name <'a, K: 'a, V: 'a>(
            leaf_edge: Handle<NodeRef<marker::Immut<'a>, K, V, marker::Leaf>, marker::Edge>,
        ) -> (Handle<NodeRef<marker::Immut<'a>, K, V, marker::Leaf>, marker::Edge>, &'a K, &'a V) {
            let mut cur_handle = match leaf_edge.$next_kv() {
                Ok(leaf_kv) => {
                    let (k, v) = leaf_kv.into_kv();
                    let next_leaf_edge = leaf_kv.$next_edge();
                    return (next_leaf_edge, k, v);
                }
                Err(last_edge) => {
                    let next_level = last_edge.into_node().ascend().ok();
                    unwrap_unchecked(next_level)
                }
            };

            loop {
                cur_handle = match cur_handle.$next_kv() {
                    Ok(internal_kv) => {
                        let (k, v) = internal_kv.into_kv();
                        let next_internal_edge = internal_kv.$next_edge();
                        let next_leaf_edge = next_internal_edge.descend().$initial_leaf_edge();
                        return (next_leaf_edge, k, v);
                    }
                    Err(last_edge) => {
                        let next_level = last_edge.into_node().ascend().ok();
                        unwrap_unchecked(next_level)
                    }
                }
            }
        }
    };
}

macro_rules! def_next_mut {
    { unsafe fn $name:ident : $next_kv:ident $next_edge:ident $initial_leaf_edge:ident } => {
        /// Given a leaf edge handle into a mutable tree, returns handles to the next
        /// leaf edge and to the KV between these edges.
        /// Unsafe for two reasons:
        /// - the caller must ensure that the given leaf edge has a successor;
        /// - both returned handles represent mutable references into the same tree
        ///   that can easily invalidate each other, even on immutable use.
        unsafe fn $name <'a, K: 'a, V: 'a>(
            leaf_edge: Handle<NodeRef<marker::Mut<'a>, K, V, marker::Leaf>, marker::Edge>,
        ) -> (Handle<NodeRef<marker::Mut<'a>, K, V, marker::Leaf>, marker::Edge>,
              Handle<NodeRef<marker::Mut<'a>, K, V, marker::LeafOrInternal>, marker::KV>) {
            let mut cur_handle = match leaf_edge.$next_kv() {
                Ok(leaf_kv) => {
                    let next_leaf_edge = ptr::read(&leaf_kv).$next_edge();
                    return (next_leaf_edge, leaf_kv.forget_node_type());
                }
                Err(last_edge) => {
                    let next_level = last_edge.into_node().ascend().ok();
                    unwrap_unchecked(next_level)
                }
            };

            loop {
                cur_handle = match cur_handle.$next_kv() {
                    Ok(internal_kv) => {
                        let next_internal_edge = ptr::read(&internal_kv).$next_edge();
                        let next_leaf_edge = next_internal_edge.descend().$initial_leaf_edge();
                        return (next_leaf_edge, internal_kv.forget_node_type());
                    }
                    Err(last_edge) => {
                        let next_level = last_edge.into_node().ascend().ok();
                        unwrap_unchecked(next_level)
                    }
                }
            }
        }
    };
}

macro_rules! def_next_dealloc {
    { unsafe fn $name:ident : $next_kv:ident $next_edge:ident $initial_leaf_edge:ident } => {
        /// Given a leaf edge handle into an owned tree, returns a handle to the next
        /// leaf edge and the key and value between these edges, while deallocating
        /// any node left behind.
        /// Unsafe for two reasons:
        /// - the caller must ensure that the given leaf edge has a successor;
        /// - the node pointed at by the given handle, and its ancestors, may be deallocated,
        ///   while the reference to those nodes in the surviving ancestors is left dangling;
        ///   thus using the returned handle is dangerous.
        unsafe fn $name <K, V>(
            leaf_edge: Handle<NodeRef<marker::Owned, K, V, marker::Leaf>, marker::Edge>,
        ) -> (Handle<NodeRef<marker::Owned, K, V, marker::Leaf>, marker::Edge>, K, V) {
            let mut cur_handle = match leaf_edge.$next_kv() {
                Ok(leaf_kv) => {
                    let k = ptr::read(leaf_kv.reborrow().into_kv().0);
                    let v = ptr::read(leaf_kv.reborrow().into_kv().1);
                    let next_leaf_edge = leaf_kv.$next_edge();
                    return (next_leaf_edge, k, v);
                }
                Err(last_edge) => {
                    unwrap_unchecked(last_edge.into_node().deallocate_and_ascend())
                }
            };

            loop {
                cur_handle = match cur_handle.$next_kv() {
                    Ok(internal_kv) => {
                        let k = ptr::read(internal_kv.reborrow().into_kv().0);
                        let v = ptr::read(internal_kv.reborrow().into_kv().1);
                        let next_internal_edge = internal_kv.$next_edge();
                        let next_leaf_edge = next_internal_edge.descend().$initial_leaf_edge();
                        return (next_leaf_edge, k, v);
                    }
                    Err(last_edge) => {
                        unwrap_unchecked(last_edge.into_node().deallocate_and_ascend())
                    }
                }
            }
        }
    };
}

def_next! {unsafe fn next_unchecked: right_kv right_edge first_leaf_edge}
def_next! {unsafe fn next_back_unchecked: left_kv left_edge last_leaf_edge}
def_next_mut! {unsafe fn next_unchecked_mut: right_kv right_edge first_leaf_edge}
def_next_mut! {unsafe fn next_back_unchecked_mut: left_kv left_edge last_leaf_edge}
def_next_dealloc! {unsafe fn next_unchecked_deallocating: right_kv right_edge first_leaf_edge}
def_next_dealloc! {unsafe fn next_back_unchecked_deallocating: left_kv left_edge last_leaf_edge}

impl<'a, K, V> Handle<NodeRef<marker::Immut<'a>, K, V, marker::Leaf>, marker::Edge> {
    /// Moves the leaf edge handle to the next leaf edge and returns references to the
    /// key and value in between.
    /// Unsafe because the caller must ensure that the leaf edge is not the last one in the tree.
    pub unsafe fn next_unchecked(&mut self) -> (&'a K, &'a V) {
        let (next_edge, k, v) = next_unchecked(*self);
        *self = next_edge;
        (k, v)
    }

    /// Moves the leaf edge handle to the previous leaf edge and returns references to the
    /// key and value in between.
    /// Unsafe because the caller must ensure that the leaf edge is not the first one in the tree.
    pub unsafe fn next_back_unchecked(&mut self) -> (&'a K, &'a V) {
        let (next_edge, k, v) = next_back_unchecked(*self);
        *self = next_edge;
        (k, v)
    }
}

impl<'a, K, V> Handle<NodeRef<marker::Mut<'a>, K, V, marker::Leaf>, marker::Edge> {
    /// Moves the leaf edge handle to the next leaf edge and returns references to the
    /// key and value in between.
    /// Unsafe for two reasons:
    /// - The caller must ensure that the leaf edge is not the last one in the tree.
    /// - Using the updated handle may well invalidate the returned references.
    pub unsafe fn next_unchecked(&mut self) -> (&'a mut K, &'a mut V) {
        let (next_edge, kv) = next_unchecked_mut(ptr::read(self));
        *self = next_edge;
        // Doing the descend (and perhaps another move) invalidates the references
        // returned by `into_kv_mut`, so we have to do this last.
        kv.into_kv_mut()
    }

    /// Moves the leaf edge handle to the previous leaf and returns references to the
    /// key and value in between.
    /// Unsafe for two reasons:
    /// - The caller must ensure that the leaf edge is not the first one in the tree.
    /// - Using the updated handle may well invalidate the returned references.
    pub unsafe fn next_back_unchecked(&mut self) -> (&'a mut K, &'a mut V) {
        let (next_edge, kv) = next_back_unchecked_mut(ptr::read(self));
        *self = next_edge;
        // Doing the descend (and perhaps another move) invalidates the references
        // returned by `into_kv_mut`, so we have to do this last.
        kv.into_kv_mut()
    }
}

impl<K, V> Handle<NodeRef<marker::Owned, K, V, marker::Leaf>, marker::Edge> {
    /// Moves the leaf edge handle to the next leaf edge and returns the key and value
    /// in between, while deallocating any node left behind.
    /// Unsafe for three reasons:
    /// - The caller must ensure that the leaf edge is not the last one in the tree
    ///   and is not a handle previously resulting from counterpart `next_back_unchecked`.
    /// - If the leaf edge is the last edge of a node, that node and possibly ancestors
    ///   will be deallocated, while the reference to those nodes in the surviving ancestor
    ///   is left dangling; thus further use of the leaf edge handle is dangerous.
    ///   It is, however, safe to call this method again on the updated handle.
    ///   if the two preconditions above hold.
    /// - Using the updated handle may well invalidate the returned references.
    pub unsafe fn next_unchecked(&mut self) -> (K, V) {
        let (next_edge, k, v) = next_unchecked_deallocating(ptr::read(self));
        *self = next_edge;
        (k, v)
    }

    /// Moves the leaf edge handle to the previous leaf edge and returns the key
    /// and value in between, while deallocating any node left behind.
    /// Unsafe for three reasons:
    /// - The caller must ensure that the leaf edge is not the first one in the tree
    ///   and is not a handle previously resulting from counterpart `next_unchecked`.
    /// - If the lead edge is the first edge of a node, that node and possibly ancestors
    ///   will be deallocated, while the reference to those nodes in the surviving ancestor
    ///   is left dangling; thus further use of the leaf edge handle is dangerous.
    ///   It is, however, safe to call this method again on the updated handle.
    ///   if the two preconditions above hold.
    /// - Using the updated handle may well invalidate the returned references.
    pub unsafe fn next_back_unchecked(&mut self) -> (K, V) {
        let (next_edge, k, v) = next_back_unchecked_deallocating(ptr::read(self));
        *self = next_edge;
        (k, v)
    }
}

impl<BorrowType, K, V> NodeRef<BorrowType, K, V, marker::LeafOrInternal> {
    /// Returns the leftmost leaf edge in or underneath a node - in other words, the edge
    /// you need first when navigating forward (or last when navigating backward).
    #[inline]
    pub fn first_leaf_edge(self) -> Handle<NodeRef<BorrowType, K, V, marker::Leaf>, marker::Edge> {
        let mut node = self;
        loop {
            match node.force() {
                Leaf(leaf) => return leaf.first_edge(),
                Internal(internal) => node = internal.first_edge().descend(),
            }
        }
    }

    /// Returns the rightmost leaf edge in or underneath a node - in other words, the edge
    /// you need last when navigating forward (or first when navigating backward).
    #[inline]
    pub fn last_leaf_edge(self) -> Handle<NodeRef<BorrowType, K, V, marker::Leaf>, marker::Edge> {
        let mut node = self;
        loop {
            match node.force() {
                Leaf(leaf) => return leaf.last_edge(),
                Internal(internal) => node = internal.last_edge().descend(),
            }
        }
    }
}
