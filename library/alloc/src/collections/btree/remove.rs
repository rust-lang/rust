use super::map::MIN_LEN;
use super::node::{marker, ForceResult::*, Handle, LeftOrRight::*, NodeRef};
use super::unwrap_unchecked;
use core::mem;

impl<'a, K: 'a, V: 'a> Handle<NodeRef<marker::Mut<'a>, K, V, marker::LeafOrInternal>, marker::KV> {
    /// Removes a key/value-pair from the tree, and returns that pair, as well as
    /// the leaf edge corresponding to that former pair. It's possible this empties
    /// a root node that is internal, which the caller should pop from the map
    /// holding the tree. The caller should also decrement the map's length.
    pub fn remove_kv_tracking<F: FnOnce()>(
        self,
        handle_emptied_internal_root: F,
    ) -> ((K, V), Handle<NodeRef<marker::Mut<'a>, K, V, marker::Leaf>, marker::Edge>) {
        match self.force() {
            Leaf(node) => node.remove_leaf_kv(handle_emptied_internal_root),
            Internal(node) => node.remove_internal_kv(handle_emptied_internal_root),
        }
    }
}

impl<'a, K: 'a, V: 'a> Handle<NodeRef<marker::Mut<'a>, K, V, marker::Leaf>, marker::KV> {
    fn remove_leaf_kv<F: FnOnce()>(
        self,
        handle_emptied_internal_root: F,
    ) -> ((K, V), Handle<NodeRef<marker::Mut<'a>, K, V, marker::Leaf>, marker::Edge>) {
        let leaf = self.reborrow().into_node();
        if leaf.len() > MIN_LEN || leaf.ascend().is_err() {
            self.remove()
        } else {
            let idx = self.idx();
            // We have to temporarily forget the child type, because there is no
            // distinct node type for the immediate parents of a leaf.
            let (old_kv, pos) = match self.into_node().forget_type().choose_parent_kv() {
                Ok(Left(left_parent_kv)) => {
                    debug_assert!(left_parent_kv.right_child_len() == MIN_LEN);
                    if left_parent_kv.can_merge_discarding_hole() {
                        left_parent_kv.merge_discarding_hole(Right(idx))
                    } else {
                        debug_assert!(left_parent_kv.left_child_len() > MIN_LEN);
                        left_parent_kv.steal_left_discarding_hole(idx)
                    }
                }
                Ok(Right(right_parent_kv)) => {
                    debug_assert!(right_parent_kv.left_child_len() == MIN_LEN);
                    if right_parent_kv.can_merge_discarding_hole() {
                        right_parent_kv.merge_discarding_hole(Left(idx))
                    } else {
                        debug_assert!(right_parent_kv.right_child_len() > MIN_LEN);
                        right_parent_kv.steal_right_discarding_hole(idx)
                    }
                }
                Err(_) => unreachable!(),
            };
            // SAFETY: `pos` is the leaf we started from or a sibling.
            let mut pos = unsafe { pos.cast_to_leaf_unchecked() };

            // Only if we merged, the parent (if any) has shrunk, but skipping
            // the following step does not pay off in benchmarks.
            //
            // SAFETY: We won't destroy or rearrange the leaf where `pos` is at
            // by handling its parent recursively; at worst we will destroy or
            // rearrange the parent through the grandparent, thus change the
            // leaf's parent pointer.
            if let Ok(parent) = unsafe { pos.reborrow_mut() }.into_node().ascend() {
                parent.into_node().handle_shrunk_node_recursively(handle_emptied_internal_root);
            }
            (old_kv, pos)
        }
    }
}

impl<'a, K: 'a, V: 'a> Handle<NodeRef<marker::Mut<'a>, K, V, marker::Internal>, marker::KV> {
    fn remove_internal_kv<F: FnOnce()>(
        self,
        handle_emptied_internal_root: F,
    ) -> ((K, V), Handle<NodeRef<marker::Mut<'a>, K, V, marker::Leaf>, marker::Edge>) {
        // Remove an adjacent KV from its leaf and then put it back in place of
        // the element we were asked to remove. Prefer the left adjacent KV,
        // for the reasons listed in `choose_parent_kv`.
        let left_leaf_kv = self.left_edge().descend().last_leaf_edge().left_kv();
        let left_leaf_kv = unsafe { unwrap_unchecked(left_leaf_kv.ok()) };
        let (left_kv, left_hole) = left_leaf_kv.remove_leaf_kv(handle_emptied_internal_root);

        // The internal node may have been stolen from or merged. Go back right
        // to find where the original KV ended up.
        let mut internal = unsafe { unwrap_unchecked(left_hole.next_kv().ok()) };
        let old_key = mem::replace(internal.kv_mut().0, left_kv.0);
        let old_val = mem::replace(internal.kv_mut().1, left_kv.1);
        let pos = internal.next_leaf_edge();
        ((old_key, old_val), pos)
    }
}

impl<'a, K: 'a, V: 'a> NodeRef<marker::Mut<'a>, K, V, marker::Internal> {
    /// Stocks up a possibly underfull internal node, recursively.
    /// Climbs up until it reaches an ancestor that has elements to spare or the root.
    fn handle_shrunk_node_recursively<F: FnOnce()>(mut self, handle_emptied_internal_root: F) {
        loop {
            self = match self.len() {
                0 => {
                    // An empty node must be the root, because length is only
                    // reduced by one, and non-root underfull nodes are stocked up,
                    // so non-root nodes never have fewer than MIN_LEN - 1 elements.
                    debug_assert!(self.ascend().is_err());
                    handle_emptied_internal_root();
                    return;
                }
                1..MIN_LEN => {
                    if let Some(parent) = self.handle_underfull_node_locally() {
                        parent
                    } else {
                        return;
                    }
                }
                _ => return,
            }
        }
    }

    /// Stocks up an underfull internal node, possibly at the cost of shrinking
    /// its parent instead, which is then returned.
    fn handle_underfull_node_locally(
        self,
    ) -> Option<NodeRef<marker::Mut<'a>, K, V, marker::Internal>> {
        match self.forget_type().choose_parent_kv() {
            Ok(Left(left_parent_kv)) => {
                debug_assert!(left_parent_kv.right_child_len() == MIN_LEN - 1);
                if left_parent_kv.can_merge() {
                    let child = left_parent_kv.merge();
                    let parent_edge = unsafe { unwrap_unchecked(child.ascend().ok()) };
                    Some(parent_edge.into_node())
                } else {
                    debug_assert!(left_parent_kv.left_child_len() > MIN_LEN);
                    left_parent_kv.steal_left();
                    None
                }
            }
            Ok(Right(right_parent_kv)) => {
                debug_assert!(right_parent_kv.left_child_len() == MIN_LEN - 1);
                if right_parent_kv.can_merge() {
                    let child = right_parent_kv.merge();
                    let parent_edge = unsafe { unwrap_unchecked(child.ascend().ok()) };
                    Some(parent_edge.into_node())
                } else {
                    debug_assert!(right_parent_kv.right_child_len() > MIN_LEN);
                    right_parent_kv.steal_right();
                    None
                }
            }
            Err(_) => None,
        }
    }
}
