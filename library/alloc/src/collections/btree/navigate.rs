use core::borrow::Borrow;
use core::ops::RangeBounds;
use core::{hint, ptr};

use super::node::ForceResult::*;
use super::node::{Handle, NodeRef, marker};
use super::search::SearchBound;
use crate::alloc::Allocator;
// `front` and `back` are always both `None` or both `Some`.
pub(super) struct LeafRange<BorrowType, K, V> {
    front: Option<Handle<NodeRef<BorrowType, K, V, marker::Leaf>, marker::Edge>>,
    back: Option<Handle<NodeRef<BorrowType, K, V, marker::Leaf>, marker::Edge>>,
}

impl<'a, K: 'a, V: 'a> Clone for LeafRange<marker::Immut<'a>, K, V> {
    fn clone(&self) -> Self {
        LeafRange { front: self.front.clone(), back: self.back.clone() }
    }
}

impl<B, K, V> Default for LeafRange<B, K, V> {
    fn default() -> Self {
        LeafRange { front: None, back: None }
    }
}

impl<BorrowType, K, V> LeafRange<BorrowType, K, V> {
    pub(super) fn none() -> Self {
        LeafRange { front: None, back: None }
    }

    fn is_empty(&self) -> bool {
        self.front == self.back
    }

    /// Temporarily takes out another, immutable equivalent of the same range.
    pub(super) fn reborrow(&self) -> LeafRange<marker::Immut<'_>, K, V> {
        LeafRange {
            front: self.front.as_ref().map(|f| f.reborrow()),
            back: self.back.as_ref().map(|b| b.reborrow()),
        }
    }
}

impl<'a, K, V> LeafRange<marker::Immut<'a>, K, V> {
    #[inline]
    pub(super) fn next_checked(&mut self) -> Option<(&'a K, &'a V)> {
        self.perform_next_checked(|kv| kv.into_kv())
    }

    #[inline]
    pub(super) fn next_back_checked(&mut self) -> Option<(&'a K, &'a V)> {
        self.perform_next_back_checked(|kv| kv.into_kv())
    }
}

impl<'a, K, V> LeafRange<marker::ValMut<'a>, K, V> {
    #[inline]
    pub(super) fn next_checked(&mut self) -> Option<(&'a K, &'a mut V)> {
        self.perform_next_checked(|kv| unsafe { ptr::read(kv) }.into_kv_valmut())
    }

    #[inline]
    pub(super) fn next_back_checked(&mut self) -> Option<(&'a K, &'a mut V)> {
        self.perform_next_back_checked(|kv| unsafe { ptr::read(kv) }.into_kv_valmut())
    }
}

impl<BorrowType: marker::BorrowType, K, V> LeafRange<BorrowType, K, V> {
    /// If possible, extract some result from the following KV and move to the edge beyond it.
    fn perform_next_checked<F, R>(&mut self, f: F) -> Option<R>
    where
        F: Fn(&Handle<NodeRef<BorrowType, K, V, marker::LeafOrInternal>, marker::KV>) -> R,
    {
        if self.is_empty() {
            None
        } else {
            super::mem::replace(self.front.as_mut().unwrap(), |front| {
                let kv = front.next_kv().ok().unwrap();
                let result = f(&kv);
                (kv.next_leaf_edge(), Some(result))
            })
        }
    }

    /// If possible, extract some result from the preceding KV and move to the edge beyond it.
    fn perform_next_back_checked<F, R>(&mut self, f: F) -> Option<R>
    where
        F: Fn(&Handle<NodeRef<BorrowType, K, V, marker::LeafOrInternal>, marker::KV>) -> R,
    {
        if self.is_empty() {
            None
        } else {
            super::mem::replace(self.back.as_mut().unwrap(), |back| {
                let kv = back.next_back_kv().ok().unwrap();
                let result = f(&kv);
                (kv.next_back_leaf_edge(), Some(result))
            })
        }
    }
}

enum LazyLeafHandle<BorrowType, K, V> {
    Root(NodeRef<BorrowType, K, V, marker::LeafOrInternal>), // not yet descended
    Edge(Handle<NodeRef<BorrowType, K, V, marker::Leaf>, marker::Edge>),
}

impl<'a, K: 'a, V: 'a> Clone for LazyLeafHandle<marker::Immut<'a>, K, V> {
    fn clone(&self) -> Self {
        match self {
            LazyLeafHandle::Root(root) => LazyLeafHandle::Root(*root),
            LazyLeafHandle::Edge(edge) => LazyLeafHandle::Edge(*edge),
        }
    }
}

impl<BorrowType, K, V> LazyLeafHandle<BorrowType, K, V> {
    fn reborrow(&self) -> LazyLeafHandle<marker::Immut<'_>, K, V> {
        match self {
            LazyLeafHandle::Root(root) => LazyLeafHandle::Root(root.reborrow()),
            LazyLeafHandle::Edge(edge) => LazyLeafHandle::Edge(edge.reborrow()),
        }
    }
}

// `front` and `back` are always both `None` or both `Some`.
pub(super) struct LazyLeafRange<BorrowType, K, V> {
    front: Option<LazyLeafHandle<BorrowType, K, V>>,
    back: Option<LazyLeafHandle<BorrowType, K, V>>,
}

impl<B, K, V> Default for LazyLeafRange<B, K, V> {
    fn default() -> Self {
        LazyLeafRange { front: None, back: None }
    }
}

impl<'a, K: 'a, V: 'a> Clone for LazyLeafRange<marker::Immut<'a>, K, V> {
    fn clone(&self) -> Self {
        LazyLeafRange { front: self.front.clone(), back: self.back.clone() }
    }
}

impl<BorrowType, K, V> LazyLeafRange<BorrowType, K, V> {
    pub(super) fn none() -> Self {
        LazyLeafRange { front: None, back: None }
    }

    /// Temporarily takes out another, immutable equivalent of the same range.
    pub(super) fn reborrow(&self) -> LazyLeafRange<marker::Immut<'_>, K, V> {
        LazyLeafRange {
            front: self.front.as_ref().map(|f| f.reborrow()),
            back: self.back.as_ref().map(|b| b.reborrow()),
        }
    }
}

impl<'a, K, V> LazyLeafRange<marker::Immut<'a>, K, V> {
    #[inline]
    pub(super) unsafe fn next_unchecked(&mut self) -> (&'a K, &'a V) {
        unsafe { self.init_front().unwrap().next_unchecked() }
    }

    #[inline]
    pub(super) unsafe fn next_back_unchecked(&mut self) -> (&'a K, &'a V) {
        unsafe { self.init_back().unwrap().next_back_unchecked() }
    }
}

impl<'a, K, V> LazyLeafRange<marker::ValMut<'a>, K, V> {
    #[inline]
    pub(super) unsafe fn next_unchecked(&mut self) -> (&'a K, &'a mut V) {
        unsafe { self.init_front().unwrap().next_unchecked() }
    }

    #[inline]
    pub(super) unsafe fn next_back_unchecked(&mut self) -> (&'a K, &'a mut V) {
        unsafe { self.init_back().unwrap().next_back_unchecked() }
    }
}

impl<K, V> LazyLeafRange<marker::Dying, K, V> {
    fn take_front(
        &mut self,
    ) -> Option<Handle<NodeRef<marker::Dying, K, V, marker::Leaf>, marker::Edge>> {
        match self.front.take()? {
            LazyLeafHandle::Root(root) => Some(root.first_leaf_edge()),
            LazyLeafHandle::Edge(edge) => Some(edge),
        }
    }

    #[inline]
    pub(super) unsafe fn deallocating_next_unchecked<A: Allocator + Clone>(
        &mut self,
        alloc: A,
    ) -> Handle<NodeRef<marker::Dying, K, V, marker::LeafOrInternal>, marker::KV> {
        debug_assert!(self.front.is_some());
        let front = self.init_front().unwrap();
        unsafe { front.deallocating_next_unchecked(alloc) }
    }

    #[inline]
    pub(super) unsafe fn deallocating_next_back_unchecked<A: Allocator + Clone>(
        &mut self,
        alloc: A,
    ) -> Handle<NodeRef<marker::Dying, K, V, marker::LeafOrInternal>, marker::KV> {
        debug_assert!(self.back.is_some());
        let back = self.init_back().unwrap();
        unsafe { back.deallocating_next_back_unchecked(alloc) }
    }

    #[inline]
    pub(super) fn deallocating_end<A: Allocator + Clone>(&mut self, alloc: A) {
        if let Some(front) = self.take_front() {
            front.deallocating_end(alloc)
        }
    }
}

impl<BorrowType: marker::BorrowType, K, V> LazyLeafRange<BorrowType, K, V> {
    fn init_front(
        &mut self,
    ) -> Option<&mut Handle<NodeRef<BorrowType, K, V, marker::Leaf>, marker::Edge>> {
        if let Some(LazyLeafHandle::Root(root)) = &self.front {
            self.front = Some(LazyLeafHandle::Edge(unsafe { ptr::read(root) }.first_leaf_edge()));
        }
        match &mut self.front {
            None => None,
            Some(LazyLeafHandle::Edge(edge)) => Some(edge),
            // SAFETY: the code above would have replaced it.
            Some(LazyLeafHandle::Root(_)) => unsafe { hint::unreachable_unchecked() },
        }
    }

    fn init_back(
        &mut self,
    ) -> Option<&mut Handle<NodeRef<BorrowType, K, V, marker::Leaf>, marker::Edge>> {
        if let Some(LazyLeafHandle::Root(root)) = &self.back {
            self.back = Some(LazyLeafHandle::Edge(unsafe { ptr::read(root) }.last_leaf_edge()));
        }
        match &mut self.back {
            None => None,
            Some(LazyLeafHandle::Edge(edge)) => Some(edge),
            // SAFETY: the code above would have replaced it.
            Some(LazyLeafHandle::Root(_)) => unsafe { hint::unreachable_unchecked() },
        }
    }
}

impl<BorrowType: marker::BorrowType, K, V> NodeRef<BorrowType, K, V, marker::LeafOrInternal> {
    /// Finds the distinct leaf edges delimiting a specified range in a tree.
    ///
    /// If such distinct edges exist, returns them in ascending order, meaning
    /// that a non-zero number of calls to `next_unchecked` on the `front` of
    /// the result and/or calls to `next_back_unchecked` on the `back` of the
    /// result will eventually reach the same edge.
    ///
    /// If there are no such edges, i.e., if the tree contains no key within
    /// the range, returns an empty `front` and `back`.
    ///
    /// # Safety
    /// Unless `BorrowType` is `Immut`, do not use the handles to visit the same
    /// KV twice.
    unsafe fn find_leaf_edges_spanning_range<Q: ?Sized, R>(
        self,
        range: R,
    ) -> LeafRange<BorrowType, K, V>
    where
        Q: Ord,
        K: Borrow<Q>,
        R: RangeBounds<Q>,
    {
        match self.search_tree_for_bifurcation(&range) {
            Err(_) => LeafRange::none(),
            Ok((
                node,
                lower_edge_idx,
                upper_edge_idx,
                mut lower_child_bound,
                mut upper_child_bound,
            )) => {
                let mut lower_edge = unsafe { Handle::new_edge(ptr::read(&node), lower_edge_idx) };
                let mut upper_edge = unsafe { Handle::new_edge(node, upper_edge_idx) };
                loop {
                    match (lower_edge.force(), upper_edge.force()) {
                        (Leaf(f), Leaf(b)) => return LeafRange { front: Some(f), back: Some(b) },
                        (Internal(f), Internal(b)) => {
                            (lower_edge, lower_child_bound) =
                                f.descend().find_lower_bound_edge(lower_child_bound);
                            (upper_edge, upper_child_bound) =
                                b.descend().find_upper_bound_edge(upper_child_bound);
                        }
                        _ => unreachable!("BTreeMap has different depths"),
                    }
                }
            }
        }
    }
}

fn full_range<BorrowType: marker::BorrowType, K, V>(
    root1: NodeRef<BorrowType, K, V, marker::LeafOrInternal>,
    root2: NodeRef<BorrowType, K, V, marker::LeafOrInternal>,
) -> LazyLeafRange<BorrowType, K, V> {
    LazyLeafRange {
        front: Some(LazyLeafHandle::Root(root1)),
        back: Some(LazyLeafHandle::Root(root2)),
    }
}

impl<'a, K: 'a, V: 'a> NodeRef<marker::Immut<'a>, K, V, marker::LeafOrInternal> {
    /// Finds the pair of leaf edges delimiting a specific range in a tree.
    ///
    /// The result is meaningful only if the tree is ordered by key, like the tree
    /// in a `BTreeMap` is.
    pub(super) fn range_search<Q, R>(self, range: R) -> LeafRange<marker::Immut<'a>, K, V>
    where
        Q: ?Sized + Ord,
        K: Borrow<Q>,
        R: RangeBounds<Q>,
    {
        // SAFETY: our borrow type is immutable.
        unsafe { self.find_leaf_edges_spanning_range(range) }
    }

    /// Finds the pair of leaf edges delimiting an entire tree.
    pub(super) fn full_range(self) -> LazyLeafRange<marker::Immut<'a>, K, V> {
        full_range(self, self)
    }
}

impl<'a, K: 'a, V: 'a> NodeRef<marker::ValMut<'a>, K, V, marker::LeafOrInternal> {
    /// Splits a unique reference into a pair of leaf edges delimiting a specified range.
    /// The result are non-unique references allowing (some) mutation, which must be used
    /// carefully.
    ///
    /// The result is meaningful only if the tree is ordered by key, like the tree
    /// in a `BTreeMap` is.
    ///
    /// # Safety
    /// Do not use the duplicate handles to visit the same KV twice.
    pub(super) fn range_search<Q, R>(self, range: R) -> LeafRange<marker::ValMut<'a>, K, V>
    where
        Q: ?Sized + Ord,
        K: Borrow<Q>,
        R: RangeBounds<Q>,
    {
        unsafe { self.find_leaf_edges_spanning_range(range) }
    }

    /// Splits a unique reference into a pair of leaf edges delimiting the full range of the tree.
    /// The results are non-unique references allowing mutation (of values only), so must be used
    /// with care.
    pub(super) fn full_range(self) -> LazyLeafRange<marker::ValMut<'a>, K, V> {
        // We duplicate the root NodeRef here -- we will never visit the same KV
        // twice, and never end up with overlapping value references.
        let self2 = unsafe { ptr::read(&self) };
        full_range(self, self2)
    }
}

impl<K, V> NodeRef<marker::Dying, K, V, marker::LeafOrInternal> {
    /// Splits a unique reference into a pair of leaf edges delimiting the full range of the tree.
    /// The results are non-unique references allowing massively destructive mutation, so must be
    /// used with the utmost care.
    pub(super) fn full_range(self) -> LazyLeafRange<marker::Dying, K, V> {
        // We duplicate the root NodeRef here -- we will never access it in a way
        // that overlaps references obtained from the root.
        let self2 = unsafe { ptr::read(&self) };
        full_range(self, self2)
    }
}

impl<BorrowType: marker::BorrowType, K, V>
    Handle<NodeRef<BorrowType, K, V, marker::Leaf>, marker::Edge>
{
    /// Given a leaf edge handle, returns [`Result::Ok`] with a handle to the neighboring KV
    /// on the right side, which is either in the same leaf node or in an ancestor node.
    /// If the leaf edge is the last one in the tree, returns [`Result::Err`] with the root node.
    pub(super) fn next_kv(
        self,
    ) -> Result<
        Handle<NodeRef<BorrowType, K, V, marker::LeafOrInternal>, marker::KV>,
        NodeRef<BorrowType, K, V, marker::LeafOrInternal>,
    > {
        let mut edge = self.forget_node_type();
        loop {
            edge = match edge.right_kv() {
                Ok(kv) => return Ok(kv),
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
    pub(super) fn next_back_kv(
        self,
    ) -> Result<
        Handle<NodeRef<BorrowType, K, V, marker::LeafOrInternal>, marker::KV>,
        NodeRef<BorrowType, K, V, marker::LeafOrInternal>,
    > {
        let mut edge = self.forget_node_type();
        loop {
            edge = match edge.left_kv() {
                Ok(kv) => return Ok(kv),
                Err(last_edge) => match last_edge.into_node().ascend() {
                    Ok(parent_edge) => parent_edge.forget_node_type(),
                    Err(root) => return Err(root),
                },
            }
        }
    }
}

impl<BorrowType: marker::BorrowType, K, V>
    Handle<NodeRef<BorrowType, K, V, marker::Internal>, marker::Edge>
{
    /// Given an internal edge handle, returns [`Result::Ok`] with a handle to the neighboring KV
    /// on the right side, which is either in the same internal node or in an ancestor node.
    /// If the internal edge is the last one in the tree, returns [`Result::Err`] with the root node.
    fn next_kv(
        self,
    ) -> Result<
        Handle<NodeRef<BorrowType, K, V, marker::Internal>, marker::KV>,
        NodeRef<BorrowType, K, V, marker::Internal>,
    > {
        let mut edge = self;
        loop {
            edge = match edge.right_kv() {
                Ok(internal_kv) => return Ok(internal_kv),
                Err(last_edge) => match last_edge.into_node().ascend() {
                    Ok(parent_edge) => parent_edge,
                    Err(root) => return Err(root),
                },
            }
        }
    }
}

impl<K, V> Handle<NodeRef<marker::Dying, K, V, marker::Leaf>, marker::Edge> {
    /// Given a leaf edge handle into a dying tree, returns the next leaf edge
    /// on the right side, and the key-value pair in between, if they exist.
    ///
    /// If the given edge is the last one in a leaf, this method deallocates
    /// the leaf, as well as any ancestor nodes whose last edge was reached.
    /// This implies that if no more key-value pair follows, the entire tree
    /// will have been deallocated and there is nothing left to return.
    ///
    /// # Safety
    /// - The given edge must not have been previously returned by counterpart
    ///   `deallocating_next_back`.
    /// - The returned KV handle is only valid to access the key and value,
    ///   and only valid until the next call to a `deallocating_` method.
    unsafe fn deallocating_next<A: Allocator + Clone>(
        self,
        alloc: A,
    ) -> Option<(Self, Handle<NodeRef<marker::Dying, K, V, marker::LeafOrInternal>, marker::KV>)>
    {
        let mut edge = self.forget_node_type();
        loop {
            edge = match edge.right_kv() {
                Ok(kv) => return Some((unsafe { ptr::read(&kv) }.next_leaf_edge(), kv)),
                Err(last_edge) => {
                    match unsafe { last_edge.into_node().deallocate_and_ascend(alloc.clone()) } {
                        Some(parent_edge) => parent_edge.forget_node_type(),
                        None => return None,
                    }
                }
            }
        }
    }

    /// Given a leaf edge handle into a dying tree, returns the next leaf edge
    /// on the left side, and the key-value pair in between, if they exist.
    ///
    /// If the given edge is the first one in a leaf, this method deallocates
    /// the leaf, as well as any ancestor nodes whose first edge was reached.
    /// This implies that if no more key-value pair follows, the entire tree
    /// will have been deallocated and there is nothing left to return.
    ///
    /// # Safety
    /// - The given edge must not have been previously returned by counterpart
    ///   `deallocating_next`.
    /// - The returned KV handle is only valid to access the key and value,
    ///   and only valid until the next call to a `deallocating_` method.
    unsafe fn deallocating_next_back<A: Allocator + Clone>(
        self,
        alloc: A,
    ) -> Option<(Self, Handle<NodeRef<marker::Dying, K, V, marker::LeafOrInternal>, marker::KV>)>
    {
        let mut edge = self.forget_node_type();
        loop {
            edge = match edge.left_kv() {
                Ok(kv) => return Some((unsafe { ptr::read(&kv) }.next_back_leaf_edge(), kv)),
                Err(last_edge) => {
                    match unsafe { last_edge.into_node().deallocate_and_ascend(alloc.clone()) } {
                        Some(parent_edge) => parent_edge.forget_node_type(),
                        None => return None,
                    }
                }
            }
        }
    }

    /// Deallocates a pile of nodes from the leaf up to the root.
    /// This is the only way to deallocate the remainder of a tree after
    /// `deallocating_next` and `deallocating_next_back` have been nibbling at
    /// both sides of the tree, and have hit the same edge. As it is intended
    /// only to be called when all keys and values have been returned,
    /// no cleanup is done on any of the keys or values.
    fn deallocating_end<A: Allocator + Clone>(self, alloc: A) {
        let mut edge = self.forget_node_type();
        while let Some(parent_edge) =
            unsafe { edge.into_node().deallocate_and_ascend(alloc.clone()) }
        {
            edge = parent_edge.forget_node_type();
        }
    }
}

impl<'a, K, V> Handle<NodeRef<marker::Immut<'a>, K, V, marker::Leaf>, marker::Edge> {
    /// Moves the leaf edge handle to the next leaf edge and returns references to the
    /// key and value in between.
    ///
    /// # Safety
    /// There must be another KV in the direction travelled.
    unsafe fn next_unchecked(&mut self) -> (&'a K, &'a V) {
        super::mem::replace(self, |leaf_edge| {
            let kv = leaf_edge.next_kv().ok().unwrap();
            (kv.next_leaf_edge(), kv.into_kv())
        })
    }

    /// Moves the leaf edge handle to the previous leaf edge and returns references to the
    /// key and value in between.
    ///
    /// # Safety
    /// There must be another KV in the direction travelled.
    unsafe fn next_back_unchecked(&mut self) -> (&'a K, &'a V) {
        super::mem::replace(self, |leaf_edge| {
            let kv = leaf_edge.next_back_kv().ok().unwrap();
            (kv.next_back_leaf_edge(), kv.into_kv())
        })
    }
}

impl<'a, K, V> Handle<NodeRef<marker::ValMut<'a>, K, V, marker::Leaf>, marker::Edge> {
    /// Moves the leaf edge handle to the next leaf edge and returns references to the
    /// key and value in between.
    ///
    /// # Safety
    /// There must be another KV in the direction travelled.
    unsafe fn next_unchecked(&mut self) -> (&'a K, &'a mut V) {
        let kv = super::mem::replace(self, |leaf_edge| {
            let kv = leaf_edge.next_kv().ok().unwrap();
            (unsafe { ptr::read(&kv) }.next_leaf_edge(), kv)
        });
        // Doing this last is faster, according to benchmarks.
        kv.into_kv_valmut()
    }

    /// Moves the leaf edge handle to the previous leaf and returns references to the
    /// key and value in between.
    ///
    /// # Safety
    /// There must be another KV in the direction travelled.
    unsafe fn next_back_unchecked(&mut self) -> (&'a K, &'a mut V) {
        let kv = super::mem::replace(self, |leaf_edge| {
            let kv = leaf_edge.next_back_kv().ok().unwrap();
            (unsafe { ptr::read(&kv) }.next_back_leaf_edge(), kv)
        });
        // Doing this last is faster, according to benchmarks.
        kv.into_kv_valmut()
    }
}

impl<K, V> Handle<NodeRef<marker::Dying, K, V, marker::Leaf>, marker::Edge> {
    /// Moves the leaf edge handle to the next leaf edge and returns the key and value
    /// in between, deallocating any node left behind while leaving the corresponding
    /// edge in its parent node dangling.
    ///
    /// # Safety
    /// - There must be another KV in the direction travelled.
    /// - That KV was not previously returned by counterpart
    ///   `deallocating_next_back_unchecked` on any copy of the handles
    ///   being used to traverse the tree.
    ///
    /// The only safe way to proceed with the updated handle is to compare it, drop it,
    /// or call this method or counterpart `deallocating_next_back_unchecked` again.
    unsafe fn deallocating_next_unchecked<A: Allocator + Clone>(
        &mut self,
        alloc: A,
    ) -> Handle<NodeRef<marker::Dying, K, V, marker::LeafOrInternal>, marker::KV> {
        super::mem::replace(self, |leaf_edge| unsafe {
            leaf_edge.deallocating_next(alloc).unwrap()
        })
    }

    /// Moves the leaf edge handle to the previous leaf edge and returns the key and value
    /// in between, deallocating any node left behind while leaving the corresponding
    /// edge in its parent node dangling.
    ///
    /// # Safety
    /// - There must be another KV in the direction travelled.
    /// - That leaf edge was not previously returned by counterpart
    ///   `deallocating_next_unchecked` on any copy of the handles
    ///   being used to traverse the tree.
    ///
    /// The only safe way to proceed with the updated handle is to compare it, drop it,
    /// or call this method or counterpart `deallocating_next_unchecked` again.
    unsafe fn deallocating_next_back_unchecked<A: Allocator + Clone>(
        &mut self,
        alloc: A,
    ) -> Handle<NodeRef<marker::Dying, K, V, marker::LeafOrInternal>, marker::KV> {
        super::mem::replace(self, |leaf_edge| unsafe {
            leaf_edge.deallocating_next_back(alloc).unwrap()
        })
    }
}

impl<BorrowType: marker::BorrowType, K, V> NodeRef<BorrowType, K, V, marker::LeafOrInternal> {
    /// Returns the leftmost leaf edge in or underneath a node - in other words, the edge
    /// you need first when navigating forward (or last when navigating backward).
    #[inline]
    pub(super) fn first_leaf_edge(
        self,
    ) -> Handle<NodeRef<BorrowType, K, V, marker::Leaf>, marker::Edge> {
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
    pub(super) fn last_leaf_edge(
        self,
    ) -> Handle<NodeRef<BorrowType, K, V, marker::Leaf>, marker::Edge> {
        let mut node = self;
        loop {
            match node.force() {
                Leaf(leaf) => return leaf.last_edge(),
                Internal(internal) => node = internal.last_edge().descend(),
            }
        }
    }
}

pub(super) enum Position<BorrowType, K, V> {
    Leaf(NodeRef<BorrowType, K, V, marker::Leaf>),
    Internal(NodeRef<BorrowType, K, V, marker::Internal>),
    InternalKV,
}

impl<'a, K: 'a, V: 'a> NodeRef<marker::Immut<'a>, K, V, marker::LeafOrInternal> {
    /// Visits leaf nodes and internal KVs in order of ascending keys, and also
    /// visits internal nodes as a whole in a depth first order, meaning that
    /// internal nodes precede their individual KVs and their child nodes.
    pub(super) fn visit_nodes_in_order<F>(self, mut visit: F)
    where
        F: FnMut(Position<marker::Immut<'a>, K, V>),
    {
        match self.force() {
            Leaf(leaf) => visit(Position::Leaf(leaf)),
            Internal(internal) => {
                visit(Position::Internal(internal));
                let mut edge = internal.first_edge();
                loop {
                    edge = match edge.descend().force() {
                        Leaf(leaf) => {
                            visit(Position::Leaf(leaf));
                            match edge.next_kv() {
                                Ok(kv) => {
                                    visit(Position::InternalKV);
                                    kv.right_edge()
                                }
                                Err(_) => return,
                            }
                        }
                        Internal(internal) => {
                            visit(Position::Internal(internal));
                            internal.first_edge()
                        }
                    }
                }
            }
        }
    }

    /// Calculates the number of elements in a (sub)tree.
    pub(super) fn calc_length(self) -> usize {
        let mut result = 0;
        self.visit_nodes_in_order(|pos| match pos {
            Position::Leaf(node) => result += node.len(),
            Position::Internal(node) => result += node.len(),
            Position::InternalKV => (),
        });
        result
    }
}

impl<BorrowType: marker::BorrowType, K, V>
    Handle<NodeRef<BorrowType, K, V, marker::LeafOrInternal>, marker::KV>
{
    /// Returns the leaf edge closest to a KV for forward navigation.
    pub(super) fn next_leaf_edge(
        self,
    ) -> Handle<NodeRef<BorrowType, K, V, marker::Leaf>, marker::Edge> {
        match self.force() {
            Leaf(leaf_kv) => leaf_kv.right_edge(),
            Internal(internal_kv) => {
                let next_internal_edge = internal_kv.right_edge();
                next_internal_edge.descend().first_leaf_edge()
            }
        }
    }

    /// Returns the leaf edge closest to a KV for backward navigation.
    pub(super) fn next_back_leaf_edge(
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

impl<BorrowType: marker::BorrowType, K, V> NodeRef<BorrowType, K, V, marker::LeafOrInternal> {
    /// Returns the leaf edge corresponding to the first point at which the
    /// given bound is true.
    pub(super) fn lower_bound<Q: ?Sized>(
        self,
        mut bound: SearchBound<&Q>,
    ) -> Handle<NodeRef<BorrowType, K, V, marker::Leaf>, marker::Edge>
    where
        Q: Ord,
        K: Borrow<Q>,
    {
        let mut node = self;
        loop {
            let (edge, new_bound) = node.find_lower_bound_edge(bound);
            match edge.force() {
                Leaf(edge) => return edge,
                Internal(edge) => {
                    node = edge.descend();
                    bound = new_bound;
                }
            }
        }
    }

    /// Returns the leaf edge corresponding to the last point at which the
    /// given bound is true.
    pub(super) fn upper_bound<Q: ?Sized>(
        self,
        mut bound: SearchBound<&Q>,
    ) -> Handle<NodeRef<BorrowType, K, V, marker::Leaf>, marker::Edge>
    where
        Q: Ord,
        K: Borrow<Q>,
    {
        let mut node = self;
        loop {
            let (edge, new_bound) = node.find_upper_bound_edge(bound);
            match edge.force() {
                Leaf(edge) => return edge,
                Internal(edge) => {
                    node = edge.descend();
                    bound = new_bound;
                }
            }
        }
    }
}
