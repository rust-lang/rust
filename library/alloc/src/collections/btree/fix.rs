use super::map::MIN_LEN;
use super::node::{marker, ForceResult::*, Handle, LeftOrRight::*, NodeRef, Root};

impl<'a, K: 'a, V: 'a> NodeRef<marker::Mut<'a>, K, V, marker::LeafOrInternal> {
    /// Stocks up a possibly underfull node by merging with or stealing from a
    /// sibling. If successful but at the cost of shrinking the parent node,
    /// returns that shrunk parent node. Returns an `Err` if the node is
    /// an empty root.
    fn fix_node_through_parent(
        self,
    ) -> Result<Option<NodeRef<marker::Mut<'a>, K, V, marker::Internal>>, Self> {
        let len = self.len();
        if len >= MIN_LEN {
            Ok(None)
        } else {
            match self.choose_parent_kv() {
                Ok(Left(mut left_parent_kv)) => {
                    if left_parent_kv.can_merge() {
                        let parent = left_parent_kv.merge_tracking_parent();
                        Ok(Some(parent))
                    } else {
                        left_parent_kv.bulk_steal_left(MIN_LEN - len);
                        Ok(None)
                    }
                }
                Ok(Right(mut right_parent_kv)) => {
                    if right_parent_kv.can_merge() {
                        let parent = right_parent_kv.merge_tracking_parent();
                        Ok(Some(parent))
                    } else {
                        right_parent_kv.bulk_steal_right(MIN_LEN - len);
                        Ok(None)
                    }
                }
                Err(root) => {
                    if len > 0 {
                        Ok(None)
                    } else {
                        Err(root)
                    }
                }
            }
        }
    }
}

impl<'a, K: 'a, V: 'a> NodeRef<marker::Mut<'a>, K, V, marker::LeafOrInternal> {
    /// Stocks up a possibly underfull node, and if that causes its parent node
    /// to shrink, stocks up the parent, recursively.
    /// Returns `true` if it fixed the tree, `false` if it couldn't because the
    /// root node became empty.
    ///
    /// This method does not expect ancestors to already be underfull upon entry
    /// and panics if it encounters an empty ancestor.
    pub fn fix_node_and_affected_ancestors(mut self) -> bool {
        loop {
            match self.fix_node_through_parent() {
                Ok(Some(parent)) => self = parent.forget_type(),
                Ok(None) => return true,
                Err(_) => return false,
            }
        }
    }
}

impl<K, V> Root<K, V> {
    /// Removes empty levels on the top, but keeps an empty leaf if the entire tree is empty.
    pub fn fix_top(&mut self) {
        while self.height() > 0 && self.len() == 0 {
            self.pop_internal_level();
        }
    }

    /// Stocks up or merge away any underfull nodes on the right border of the
    /// tree. The other nodes, those that are not the root nor a rightmost edge,
    /// must already have at least MIN_LEN elements.
    pub fn fix_right_border(&mut self) {
        self.fix_top();
        if self.len() > 0 {
            self.borrow_mut().last_kv().fix_right_border_of_right_edge();
            self.fix_top();
        }
    }

    /// The symmetric clone of `fix_right_border`.
    pub fn fix_left_border(&mut self) {
        self.fix_top();
        if self.len() > 0 {
            self.borrow_mut().first_kv().fix_left_border_of_left_edge();
            self.fix_top();
        }
    }

    /// Stock up any underfull nodes on the right border of the tree.
    /// The other nodes, those that are not the root nor a rightmost edge,
    /// must be prepared to have up to MIN_LEN elements stolen.
    pub fn fix_right_border_of_plentiful(&mut self) {
        let mut cur_node = self.borrow_mut();
        while let Internal(internal) = cur_node.force() {
            // Check if right-most child is underfull.
            let mut last_kv = internal.last_kv().consider_for_balancing();
            debug_assert!(last_kv.left_child_len() >= MIN_LEN * 2);
            let right_child_len = last_kv.right_child_len();
            if right_child_len < MIN_LEN {
                // We need to steal.
                last_kv.bulk_steal_left(MIN_LEN - right_child_len);
            }

            // Go further down.
            cur_node = last_kv.into_right_child();
        }
    }
}

impl<'a, K: 'a, V: 'a> Handle<NodeRef<marker::Mut<'a>, K, V, marker::LeafOrInternal>, marker::KV> {
    fn fix_left_border_of_left_edge(mut self) {
        while let Internal(internal_kv) = self.force() {
            self = internal_kv.fix_left_child().first_kv();
            debug_assert!(self.reborrow().into_node().len() > MIN_LEN);
        }
    }

    fn fix_right_border_of_right_edge(mut self) {
        while let Internal(internal_kv) = self.force() {
            self = internal_kv.fix_right_child().last_kv();
            debug_assert!(self.reborrow().into_node().len() > MIN_LEN);
        }
    }
}

impl<'a, K: 'a, V: 'a> Handle<NodeRef<marker::Mut<'a>, K, V, marker::Internal>, marker::KV> {
    /// Stocks up the left child, assuming the right child isn't underfull, and
    /// provisions an extra element to allow merging its children in turn
    /// without becoming underfull.
    /// Returns the left child.
    fn fix_left_child(self) -> NodeRef<marker::Mut<'a>, K, V, marker::LeafOrInternal> {
        let mut internal_kv = self.consider_for_balancing();
        let left_len = internal_kv.left_child_len();
        debug_assert!(internal_kv.right_child_len() >= MIN_LEN);
        if internal_kv.can_merge() {
            internal_kv.merge_tracking_child()
        } else {
            // `MIN_LEN + 1` to avoid readjust if merge happens on the next level.
            let count = (MIN_LEN + 1).saturating_sub(left_len);
            if count > 0 {
                internal_kv.bulk_steal_right(count);
            }
            internal_kv.into_left_child()
        }
    }

    /// Stocks up the right child, assuming the left child isn't underfull, and
    /// provisions an extra element to allow merging its children in turn
    /// without becoming underfull.
    /// Returns wherever the right child ended up.
    fn fix_right_child(self) -> NodeRef<marker::Mut<'a>, K, V, marker::LeafOrInternal> {
        let mut internal_kv = self.consider_for_balancing();
        let right_len = internal_kv.right_child_len();
        debug_assert!(internal_kv.left_child_len() >= MIN_LEN);
        if internal_kv.can_merge() {
            internal_kv.merge_tracking_child()
        } else {
            // `MIN_LEN + 1` to avoid readjust if merge happens on the next level.
            let count = (MIN_LEN + 1).saturating_sub(right_len);
            if count > 0 {
                internal_kv.bulk_steal_left(count);
            }
            internal_kv.into_right_child()
        }
    }
}
