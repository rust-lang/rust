use core::intrinsics;
use core::mem;
use core::ptr;

use super::node::{marker, ForceResult::*, Handle, NodeRef};
use super::unwrap_unchecked;

impl<BorrowType, K, V> Handle<NodeRef<BorrowType, K, V, marker::Leaf>, marker::Edge> {
    /// Given a leaf edge handle, returns [`Result::Ok`] with a handle to the neighboring KV
    /// on the right side, which is either in the same leaf node or in an ancestor node.
    /// If the leaf edge is the last one in the tree, returns [`Result::Err`] with the root node.
    pub fn next_kv(
        self,
    ) -> Result<
        Handle<NodeRef<BorrowType, K, V, marker::LeafOrInternal>, marker::KV>,
        NodeRef<BorrowType, K, V, marker::LeafOrInternal>,
    > {
        let mut edge = self.forget_node_type();
        loop {
            edge = match edge.right_kv() {
                Ok(internal_kv) => return Ok(internal_kv),
                Err(last_edge) => match last_edge.into_node().ascend() {
                    Ok(parent_edge) => parent_edge.forget_node_type(),
                    Err(root) => return Err(root),
                },
            }
        }
    }

    /// Given a leaf edge handle, returns [`Result::Ok`] with a handle to the neighboring KV
    /// on the left side, which is either in the same leaf node or in an ancestor node.
    /// If the leaf edge is the first one in the tree, returns [`Result::Err`] with the root node.
    pub fn next_back_kv(
        self,
    ) -> Result<
        Handle<NodeRef<BorrowType, K, V, marker::LeafOrInternal>, marker::KV>,
        NodeRef<BorrowType, K, V, marker::LeafOrInternal>,
    > {
        let mut edge = self.forget_node_type();
        loop {
            edge = match edge.left_kv() {
                Ok(internal_kv) => return Ok(internal_kv),
                Err(last_edge) => match last_edge.into_node().ascend() {
                    Ok(parent_edge) => parent_edge.forget_node_type(),
                    Err(root) => return Err(root),
                },
            }
        }
    }
}

macro_rules! def_next_kv_uncheched_dealloc {
    { unsafe fn $name:ident : $adjacent_kv:ident } => {
        /// Given a leaf edge handle into an owned tree, returns a handle to the next KV,
        /// while deallocating any node left behind.
        /// Unsafe for two reasons:
        /// - The caller must ensure that the leaf edge is not the last one in the tree.
        /// - The node pointed at by the given handle, and its ancestors, may be deallocated,
        ///   while the reference to those nodes in the surviving ancestors is left dangling;
        ///   thus using the returned handle to navigate further is dangerous.
        unsafe fn $name <K, V>(
            leaf_edge: Handle<NodeRef<marker::Owned, K, V, marker::Leaf>, marker::Edge>,
        ) -> Handle<NodeRef<marker::Owned, K, V, marker::LeafOrInternal>, marker::KV> {
            let mut edge = leaf_edge.forget_node_type();
            loop {
                edge = match edge.$adjacent_kv() {
                    Ok(internal_kv) => return internal_kv,
                    Err(last_edge) => {
                        unsafe {
                            let parent_edge = last_edge.into_node().deallocate_and_ascend();
                            unwrap_unchecked(parent_edge).forget_node_type()
                        }
                    }
                }
            }
        }
    };
}

def_next_kv_uncheched_dealloc! {unsafe fn next_kv_unchecked_dealloc: right_kv}
def_next_kv_uncheched_dealloc! {unsafe fn next_back_kv_unchecked_dealloc: left_kv}

/// This replaces the value behind the `v` unique reference by calling the
/// relevant function, and returns a result obtained along the way.
///
/// If a panic occurs in the `change` closure, the entire process will be aborted.
#[inline]
fn replace<T, R>(v: &mut T, change: impl FnOnce(T) -> (T, R)) -> R {
    struct PanicGuard;
    impl Drop for PanicGuard {
        fn drop(&mut self) {
            intrinsics::abort()
        }
    }
    let guard = PanicGuard;
    let value = unsafe { ptr::read(v) };
    let (new_value, ret) = change(value);
    unsafe {
        ptr::write(v, new_value);
    }
    mem::forget(guard);
    ret
}

impl<'a, K, V> Handle<NodeRef<marker::Immut<'a>, K, V, marker::Leaf>, marker::Edge> {
    /// Moves the leaf edge handle to the next leaf edge and returns references to the
    /// key and value in between.
    /// Unsafe because the caller must ensure that the leaf edge is not the last one in the tree.
    pub unsafe fn next_unchecked(&mut self) -> (&'a K, &'a V) {
        replace(self, |leaf_edge| {
            let kv = leaf_edge.next_kv();
            let kv = unsafe { unwrap_unchecked(kv.ok()) };
            (kv.next_leaf_edge(), kv.into_kv())
        })
    }

    /// Moves the leaf edge handle to the previous leaf edge and returns references to the
    /// key and value in between.
    /// Unsafe because the caller must ensure that the leaf edge is not the first one in the tree.
    pub unsafe fn next_back_unchecked(&mut self) -> (&'a K, &'a V) {
        replace(self, |leaf_edge| {
            let kv = leaf_edge.next_back_kv();
            let kv = unsafe { unwrap_unchecked(kv.ok()) };
            (kv.next_back_leaf_edge(), kv.into_kv())
        })
    }
}

impl<'a, K, V> Handle<NodeRef<marker::Mut<'a>, K, V, marker::Leaf>, marker::Edge> {
    /// Moves the leaf edge handle to the next leaf edge and returns references to the
    /// key and value in between.
    /// Unsafe for two reasons:
    /// - The caller must ensure that the leaf edge is not the last one in the tree.
    /// - Using the updated handle may well invalidate the returned references.
    pub unsafe fn next_unchecked(&mut self) -> (&'a mut K, &'a mut V) {
        let kv = replace(self, |leaf_edge| {
            let kv = leaf_edge.next_kv();
            let kv = unsafe { unwrap_unchecked(kv.ok()) };
            (unsafe { ptr::read(&kv) }.next_leaf_edge(), kv)
        });
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
        let kv = replace(self, |leaf_edge| {
            let kv = leaf_edge.next_back_kv();
            let kv = unsafe { unwrap_unchecked(kv.ok()) };
            (unsafe { ptr::read(&kv) }.next_back_leaf_edge(), kv)
        });
        // Doing the descend (and perhaps another move) invalidates the references
        // returned by `into_kv_mut`, so we have to do this last.
        kv.into_kv_mut()
    }
}

impl<K, V> Handle<NodeRef<marker::Owned, K, V, marker::Leaf>, marker::Edge> {
    /// Moves the leaf edge handle to the next leaf edge and returns the key and value
    /// in between, while deallocating any node left behind.
    /// Unsafe for two reasons:
    /// - The caller must ensure that the leaf edge is not the last one in the tree
    ///   and is not a handle previously resulting from counterpart `next_back_unchecked`.
    /// - Further use of the updated leaf edge handle is very dangerous. In particular,
    ///   if the leaf edge is the last edge of a node, that node and possibly ancestors
    ///   will be deallocated, while the reference to those nodes in the surviving ancestor
    ///   is left dangling.
    ///   The only safe way to proceed with the updated handle is to compare it, drop it,
    ///   call this method again subject to both preconditions listed in the first point,
    ///   or call counterpart `next_back_unchecked` subject to its preconditions.
    pub unsafe fn next_unchecked(&mut self) -> (K, V) {
        replace(self, |leaf_edge| {
            let kv = unsafe { next_kv_unchecked_dealloc(leaf_edge) };
            let k = unsafe { ptr::read(kv.reborrow().into_kv().0) };
            let v = unsafe { ptr::read(kv.reborrow().into_kv().1) };
            (kv.next_leaf_edge(), (k, v))
        })
    }

    /// Moves the leaf edge handle to the previous leaf edge and returns the key
    /// and value in between, while deallocating any node left behind.
    /// Unsafe for two reasons:
    /// - The caller must ensure that the leaf edge is not the first one in the tree
    ///   and is not a handle previously resulting from counterpart `next_unchecked`.
    /// - Further use of the updated leaf edge handle is very dangerous. In particular,
    ///   if the leaf edge is the first edge of a node, that node and possibly ancestors
    ///   will be deallocated, while the reference to those nodes in the surviving ancestor
    ///   is left dangling.
    ///   The only safe way to proceed with the updated handle is to compare it, drop it,
    ///   call this method again subject to both preconditions listed in the first point,
    ///   or call counterpart `next_unchecked` subject to its preconditions.
    pub unsafe fn next_back_unchecked(&mut self) -> (K, V) {
        replace(self, |leaf_edge| {
            let kv = unsafe { next_back_kv_unchecked_dealloc(leaf_edge) };
            let k = unsafe { ptr::read(kv.reborrow().into_kv().0) };
            let v = unsafe { ptr::read(kv.reborrow().into_kv().1) };
            (kv.next_back_leaf_edge(), (k, v))
        })
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

impl<BorrowType, K, V> Handle<NodeRef<BorrowType, K, V, marker::LeafOrInternal>, marker::KV> {
    /// Returns the leaf edge closest to a KV for forward navigation.
    pub fn next_leaf_edge(self) -> Handle<NodeRef<BorrowType, K, V, marker::Leaf>, marker::Edge> {
        match self.force() {
            Leaf(leaf_kv) => leaf_kv.right_edge(),
            Internal(internal_kv) => {
                let next_internal_edge = internal_kv.right_edge();
                next_internal_edge.descend().first_leaf_edge()
            }
        }
    }

    /// Returns the leaf edge closest to a KV for backward navigation.
    pub fn next_back_leaf_edge(
        self,
    ) -> Handle<NodeRef<BorrowType, K, V, marker::Leaf>, marker::Edge> {
        match self.force() {
            Leaf(leaf_kv) => leaf_kv.left_edge(),
            Internal(internal_kv) => {
                let next_internal_edge = internal_kv.left_edge();
                next_internal_edge.descend().last_leaf_edge()
            }
        }
    }
}
