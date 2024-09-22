use core::alloc::Allocator;

use super::map::MIN_LEN;
use super::node::ForceResult::*;
use super::node::LeftOrRight::*;
use super::node::{Handle, NodeRef, marker};

impl<'a, K: 'a, V: 'a> Handle<NodeRef<marker::Mut<'a>, K, V, marker::LeafOrInternal>, marker::KV> {
    /// Removes a key-value pair from the tree, and returns that pair, as well as
    /// the leaf edge corresponding to that former pair. It's possible this empties
    /// a root node that is internal, which the caller should pop from the map
    /// holding the tree. The caller should also decrement the map's length.
    pub fn remove_kv_tracking<F: FnOnce(), A: Allocator + Clone>(
        self,
        handle_emptied_internal_root: F,
        alloc: A,
    ) -> ((K, V), Handle<NodeRef<marker::Mut<'a>, K, V, marker::Leaf>, marker::Edge>) {
        match self.force() {
            Leaf(node) => node.remove_leaf_kv(handle_emptied_internal_root, alloc),
            Internal(node) => node.remove_internal_kv(handle_emptied_internal_root, alloc),
        }
    }
}

impl<'a, K: 'a, V: 'a> Handle<NodeRef<marker::Mut<'a>, K, V, marker::Leaf>, marker::KV> {
    fn remove_leaf_kv<F: FnOnce(), A: Allocator + Clone>(
        self,
        handle_emptied_internal_root: F,
        alloc: A,
    ) -> ((K, V), Handle<NodeRef<marker::Mut<'a>, K, V, marker::Leaf>, marker::Edge>) {
        let (old_kv, mut pos) = self.remove();
        let len = pos.reborrow().into_node().len();
        if len < MIN_LEN {
            let idx = pos.idx();
            // We have to temporarily forget the child type, because there is no
            // distinct node type for the immediate parents of a leaf.
            let new_pos = match pos.into_node().forget_type().choose_parent_kv() {
                Ok(Left(left_parent_kv)) => {
                    debug_assert!(left_parent_kv.right_child_len() == MIN_LEN - 1);
                    if left_parent_kv.can_merge() {
                        left_parent_kv.merge_tracking_child_edge(Right(idx), alloc.clone())
                    } else {
                        debug_assert!(left_parent_kv.left_child_len() > MIN_LEN);
                        left_parent_kv.steal_left(idx)
                    }
                }
                Ok(Right(right_parent_kv)) => {
                    debug_assert!(right_parent_kv.left_child_len() == MIN_LEN - 1);
                    if right_parent_kv.can_merge() {
                        right_parent_kv.merge_tracking_child_edge(Left(idx), alloc.clone())
                    } else {
                        debug_assert!(right_parent_kv.right_child_len() > MIN_LEN);
                        right_parent_kv.steal_right(idx)
                    }
                }
                Err(pos) => unsafe { Handle::new_edge(pos, idx) },
            };
            // SAFETY: `new_pos` is the leaf we started from or a sibling.
            pos = unsafe { new_pos.cast_to_leaf_unchecked() };

            // Only if we merged, the parent (if any) has shrunk, but skipping
            // the following step otherwise does not pay off in benchmarks.
            //
            // SAFETY: We won't destroy or rearrange the leaf where `pos` is at
            // by handling its parent recursively; at worst we will destroy or
            // rearrange the parent through the grandparent, thus change the
            // link to the parent inside the leaf.
            if let Ok(parent) = unsafe { pos.reborrow_mut() }.into_node().ascend() {
                if !parent.into_node().forget_type().fix_node_and_affected_ancestors(alloc) {
                    handle_emptied_internal_root();
                }
            }
        }
        (old_kv, pos)
    }
}

impl<'a, K: 'a, V: 'a> Handle<NodeRef<marker::Mut<'a>, K, V, marker::Internal>, marker::KV> {
    fn remove_internal_kv<F: FnOnce(), A: Allocator + Clone>(
        self,
        handle_emptied_internal_root: F,
        alloc: A,
    ) -> ((K, V), Handle<NodeRef<marker::Mut<'a>, K, V, marker::Leaf>, marker::Edge>) {
        // Remove an adjacent KV from its leaf and then put it back in place of
        // the element we were asked to remove. Prefer the left adjacent KV,
        // for the reasons listed in `choose_parent_kv`.
        let left_leaf_kv = self.left_edge().descend().last_leaf_edge().left_kv();
        let left_leaf_kv = unsafe { left_leaf_kv.ok().unwrap_unchecked() };
        let (left_kv, left_hole) = left_leaf_kv.remove_leaf_kv(handle_emptied_internal_root, alloc);

        // The internal node may have been stolen from or merged. Go back right
        // to find where the original KV ended up.
        let mut internal = unsafe { left_hole.next_kv().ok().unwrap_unchecked() };
        let old_kv = internal.replace_kv(left_kv.0, left_kv.1);
        let pos = internal.next_leaf_edge();
        (old_kv, pos)
    }
}
