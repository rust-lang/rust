use super::map::MIN_LEN;
use super::node::{marker, ForceResult, Handle, NodeRef};
use super::unwrap_unchecked;
use core::mem;
use core::ptr;

impl<'a, K: 'a, V: 'a> Handle<NodeRef<marker::Mut<'a>, K, V, marker::LeafOrInternal>, marker::KV> {
    /// Removes a key/value-pair from the map, and returns that pair, as well as
    /// the leaf edge corresponding to that former pair.
    pub fn remove_kv_tracking<F: FnOnce()>(
        self,
        handle_emptied_internal_root: F,
    ) -> ((K, V), Handle<NodeRef<marker::Mut<'a>, K, V, marker::Leaf>, marker::Edge>) {
        let (old_kv, mut pos, was_internal) = match self.force() {
            ForceResult::Leaf(leaf) => {
                let (old_kv, pos) = leaf.remove();
                (old_kv, pos, false)
            }
            ForceResult::Internal(mut internal) => {
                // Replace the location freed in the internal node with an
                // adjacent KV, and remove that adjacent KV from its leaf.
                // Always choose the adjacent KV on the left side because
                // it is typically faster to pop an element from the end
                // of the KV arrays without needing to shift other elements.

                let key_loc = internal.kv_mut().0 as *mut K;
                let val_loc = internal.kv_mut().1 as *mut V;

                let to_remove = internal.left_edge().descend().last_leaf_edge().left_kv().ok();
                let to_remove = unsafe { unwrap_unchecked(to_remove) };

                let (kv, pos) = to_remove.remove();

                let old_key = unsafe { mem::replace(&mut *key_loc, kv.0) };
                let old_val = unsafe { mem::replace(&mut *val_loc, kv.1) };

                ((old_key, old_val), pos, true)
            }
        };

        // Handle underflow
        let mut cur_node = unsafe { ptr::read(&pos).into_node().forget_type() };
        let mut at_leaf = true;
        while cur_node.len() < MIN_LEN {
            match handle_underfull_node(cur_node) {
                UnderflowResult::AtRoot => break,
                UnderflowResult::Merged(edge, merged_with_left, offset) => {
                    // If we merged with our right sibling then our tracked
                    // position has not changed. However if we merged with our
                    // left sibling then our tracked position is now dangling.
                    if at_leaf && merged_with_left {
                        let idx = pos.idx() + offset;
                        let node = match unsafe { ptr::read(&edge).descend().force() } {
                            ForceResult::Leaf(leaf) => leaf,
                            ForceResult::Internal(_) => unreachable!(),
                        };
                        pos = unsafe { Handle::new_edge(node, idx) };
                    }

                    let parent = edge.into_node();
                    if parent.len() == 0 {
                        // The parent that was just emptied must be the root,
                        // because nodes on a lower level would not have been
                        // left with a single child.
                        handle_emptied_internal_root();
                        break;
                    } else {
                        cur_node = parent.forget_type();
                        at_leaf = false;
                    }
                }
                UnderflowResult::Stole(stole_from_left) => {
                    // Adjust the tracked position if we stole from a left sibling
                    if stole_from_left && at_leaf {
                        // SAFETY: This is safe since we just added an element to our node.
                        unsafe {
                            pos.move_next_unchecked();
                        }
                    }
                    break;
                }
            }
        }

        // If we deleted from an internal node then we need to compensate for
        // the earlier swap and adjust the tracked position to point to the
        // next element.
        if was_internal {
            pos = unsafe { unwrap_unchecked(pos.next_kv().ok()).next_leaf_edge() };
        }

        (old_kv, pos)
    }
}

enum UnderflowResult<'a, K, V> {
    AtRoot,
    Merged(Handle<NodeRef<marker::Mut<'a>, K, V, marker::Internal>, marker::Edge>, bool, usize),
    Stole(bool),
}

fn handle_underfull_node<'a, K: 'a, V: 'a>(
    node: NodeRef<marker::Mut<'a>, K, V, marker::LeafOrInternal>,
) -> UnderflowResult<'_, K, V> {
    let parent = match node.ascend() {
        Ok(parent) => parent,
        Err(_) => return UnderflowResult::AtRoot,
    };

    // Prefer the left KV if it exists. Merging with the left side is faster,
    // since merging happens towards the left and `node` has fewer elements.
    // Stealing from the left side is faster, since we can pop from the end of
    // the KV arrays.
    let (is_left, mut handle) = match parent.left_kv() {
        Ok(left) => (true, left),
        Err(parent) => {
            let right = unsafe { unwrap_unchecked(parent.right_kv().ok()) };
            (false, right)
        }
    };

    if handle.can_merge() {
        let offset = if is_left { handle.reborrow().left_edge().descend().len() + 1 } else { 0 };
        UnderflowResult::Merged(handle.merge(), is_left, offset)
    } else {
        if is_left {
            handle.steal_left();
        } else {
            handle.steal_right();
        }
        UnderflowResult::Stole(is_left)
    }
}
