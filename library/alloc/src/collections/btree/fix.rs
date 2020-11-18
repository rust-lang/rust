use super::map::MIN_LEN;
use super::node::{marker, ForceResult::*, Handle, LeftOrRight, LeftOrRight::*, NodeRef, Root};

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

    /// Fixes both the left border and the right border.
    pub fn fix_both_borders(&mut self) {
        self.fix_top();
        while let Internal(root_node) = self.borrow_mut().force() {
            if root_node.len() > 1 {
                self.fix_both_borders_of_multi_kv_root();
                return;
            }
            match root_node.try_moving_lone_kv() {
                LoneKvResult::MergedIntoEmptyChildren => self.pop_internal_level(),
                LoneKvResult::MergedIntoAccompaniedKvIdx(_) => {
                    self.pop_internal_level();
                    self.fix_both_borders_of_multi_kv_root();
                    return;
                }
                LoneKvResult::MovedToAccompanied(_) | LoneKvResult::Required => {
                    self.fix_both_borders_of_single_kv_root();
                    return;
                }
            }
        }
    }

    /// Fixes the right border of the left edge and the left border of the right
    /// edge, starting from a root node with a single key-value pair, i.e.,
    /// without siblings and without parent.
    pub fn fix_opposite_borders(&mut self) {
        assert!(self.len() == 1);
        while let Internal(root_node) = self.borrow_mut().force() {
            match root_node.try_moving_lone_kv() {
                LoneKvResult::MergedIntoEmptyChildren => self.pop_internal_level(),
                LoneKvResult::MergedIntoAccompaniedKvIdx(idx) => {
                    self.pop_internal_level();
                    let moved_kv = unsafe { Handle::new_kv(self.borrow_mut(), idx) };
                    let fixed = moved_kv.try_fixing_opposite_borders_and_ancestors();
                    assert!(fixed);
                    self.fix_top();
                    return;
                }
                LoneKvResult::MovedToAccompanied(moved_kv) => {
                    let fixed = moved_kv.try_fixing_opposite_borders_and_ancestors();
                    assert!(fixed);
                    self.fix_top();
                    return;
                }
                LoneKvResult::Required => {
                    self.fix_opposite_borders_of_single_kv_root();
                    return;
                }
            }
        }
    }
}

enum LoneKvResult<'a, K, V> {
    MergedIntoEmptyChildren,
    MergedIntoAccompaniedKvIdx(usize),
    MovedToAccompanied(Handle<NodeRef<marker::Mut<'a>, K, V, marker::LeafOrInternal>, marker::KV>),
    Required,
}

impl<'a, K: 'a, V: 'a> NodeRef<marker::Mut<'a>, K, V, marker::Internal> {
    /// Detects whether a root with a single key-value pair has underfull
    /// children, and tries to honor `MIN_LEN` for both children.
    /// If both children are empty, merging one level away is all we can do.
    /// If one child is empty and the other filled to capacity, merging is
    /// impossible, so we need to steal.
    fn try_moving_lone_kv(self) -> LoneKvResult<'a, K, V> {
        debug_assert!(self.len() == 1);
        let mut here = self.first_kv().consider_for_balancing();
        let (left_len, right_len) = (here.left_child_len(), here.right_child_len());
        if left_len + right_len == 0 {
            here.merge();
            LoneKvResult::MergedIntoEmptyChildren
        } else if here.can_merge() {
            here.merge();
            LoneKvResult::MergedIntoAccompaniedKvIdx(left_len)
        } else if left_len < MIN_LEN {
            let count = MIN_LEN - left_len;
            debug_assert!(right_len >= count + MIN_LEN);
            here.bulk_steal_right(count);
            let moved_kv = unsafe { Handle::new_kv(here.into_left_child(), left_len) };
            LoneKvResult::MovedToAccompanied(moved_kv)
        } else if right_len < MIN_LEN {
            let count = MIN_LEN - right_len;
            debug_assert!(left_len >= count + MIN_LEN);
            here.bulk_steal_left(count);
            let moved_kv = unsafe { Handle::new_kv(here.into_right_child(), count - 1) };
            LoneKvResult::MovedToAccompanied(moved_kv)
        } else {
            LoneKvResult::Required
        }
    }
}

impl<K, V> Root<K, V> {
    /// Fixes both borders of a root with more than one element, which implies
    /// the borders never touch.
    fn fix_both_borders_of_multi_kv_root(&mut self) {
        debug_assert!(self.len() > 1);
        self.borrow_mut().first_kv().fix_left_border_of_left_edge();
        self.borrow_mut().last_kv().fix_right_border_of_right_edge();
        self.fix_top();
    }

    /// Fixes both borders of a root with one element and non-empty children.
    fn fix_both_borders_of_single_kv_root(&mut self) {
        debug_assert!(self.len() == 1);
        if let Internal(root_node) = self.borrow_mut().force() {
            root_node.first_edge().descend().first_kv().fix_left_border_of_left_edge();
        } else {
            unreachable!()
        }
        if let Internal(root_node) = self.borrow_mut().force() {
            root_node.last_edge().descend().last_kv().fix_right_border_of_right_edge();
        } else {
            unreachable!()
        }

        // Fixing the children may have shrunk them.
        if let Internal(root_node) = self.borrow_mut().force() {
            root_node.try_moving_lone_kv();
        } else {
            unreachable!()
        }
        self.fix_top();
    }

    /// Fixes opposite borders of a root with one element and non-empty children.
    fn fix_opposite_borders_of_single_kv_root(&mut self) {
        debug_assert!(self.len() == 1);
        if let Internal(root_node) = self.borrow_mut().force() {
            root_node.first_edge().descend().last_kv().fix_right_border_of_right_edge();
        } else {
            unreachable!()
        }
        if let Internal(root_node) = self.borrow_mut().force() {
            root_node.last_edge().descend().first_kv().fix_left_border_of_left_edge();
        } else {
            unreachable!()
        }

        // Fixing the children may have shrunk them.
        if let Internal(root_node) = self.borrow_mut().force() {
            root_node.try_moving_lone_kv();
        } else {
            unreachable!()
        }
        self.fix_top();
    }
}

impl<'a, K: 'a, V: 'a> Handle<NodeRef<marker::Mut<'a>, K, V, marker::Internal>, marker::Edge> {
    /// Tries to fix the left border of the subtree, using one of the edge's
    /// adjacent KVs or the parent. Changes nothing and returns false if
    /// the edge has no adjacent KVs and no parent node.
    pub fn try_fixing_left_border_and_ancestors(mut self) -> bool {
        if unsafe { self.reborrow_mut() }.try_fixing_left_border() {
            self.into_node().forget_type().fix_node_and_affected_ancestors();
            true
        } else if let Ok(mut parent_edge) = self.into_node().ascend() {
            // Since there is no sibling, move on up to the parent,
            // an untouched incoming node always able to balance.
            let mut edge = match unsafe { parent_edge.reborrow_mut() }
                .try_fixing_child_tracking_border(Left(()))
                .ok()
                .unwrap()
                .force()
            {
                Internal(edge) => edge,
                Leaf(_) => unreachable!(),
            };
            let fixed = unsafe { edge.reborrow_mut() }.try_fixing_left_border();
            assert!(fixed);
            edge.into_node().forget_type().fix_node_through_parent().ok().unwrap();
            parent_edge.into_node().forget_type().fix_node_and_affected_ancestors();
            true
        } else {
            false
        }
    }

    fn try_fixing_left_border(self) -> bool {
        if let Ok(mut child_edge) = self.try_fixing_child_tracking_border(Left(())) {
            loop {
                child_edge = match child_edge.force() {
                    Internal(edge) => edge.try_fixing_child_tracking_border(Left(())).ok().unwrap(),
                    Leaf(_) => return true,
                }
            }
        } else {
            false
        }
    }
}

impl<'a, K: 'a, V: 'a> Handle<NodeRef<marker::Mut<'a>, K, V, marker::LeafOrInternal>, marker::KV> {
    /// Tries to fix the right border of the KV's left edge and the left border
    /// of the KV's right edge, using one or both of its sibling KVs, or using
    /// its parent. Changes nothing and returns false if the KV has neither
    /// siblings nor a parent.
    pub fn try_fixing_opposite_borders_and_ancestors(mut self) -> bool {
        if unsafe { self.reborrow_mut() }.try_fixing_opposite_borders() {
            // `self`'s KV index may be invalid now, but the node is fine.
            self.into_node().fix_node_and_affected_ancestors();
            true
        } else if let Ok(mut parent_edge) = self.into_node().ascend() {
            // Since there is no sibling, move on up to the parent,
            // an untouched incoming node always able to balance.
            let edge = unsafe { parent_edge.reborrow_mut() }
                .try_fixing_child_tracking_border(Left(()))
                .ok()
                .unwrap();
            let mut kv = edge.right_kv().ok().unwrap();
            let fixed = unsafe { kv.reborrow_mut() }.try_fixing_opposite_borders();
            assert!(fixed);
            kv.into_node().fix_node_through_parent().ok().unwrap();
            parent_edge.into_node().forget_type().fix_node_and_affected_ancestors();
            true
        } else {
            false
        }
    }

    fn try_fixing_opposite_borders(mut self) -> bool {
        let fixed_right_side;
        if let Ok(right_kv) = unsafe { self.reborrow_mut() }.right_edge().right_kv() {
            right_kv.fix_left_border_of_left_edge();
            fixed_right_side = true
        } else {
            fixed_right_side = false
        }

        // Fix on the left after fixing on the right, because we may move the KV
        // that `self` refers to. For the same reason, do not reborrow `self`.
        match self.left_edge().left_kv() {
            Ok(mut left_kv) => {
                unsafe { left_kv.reborrow_mut() }.fix_right_border_of_right_edge();
                if !fixed_right_side {
                    // Now that the left child is stocked up, we can use it to
                    // fix the rightmost child in turn.
                    left_kv.into_node().last_kv().fix_left_border_of_right_edge();
                }
                true
            }
            Err(leftmost_edge) => {
                if fixed_right_side {
                    // Since the right child was stocked up, we can use
                    // it to fix the leftmost child in turn.
                    leftmost_edge.into_node().first_kv().fix_right_border_of_left_edge();
                    true
                } else {
                    false
                }
            }
        }
    }
}

impl<'a, K: 'a, V: 'a> Handle<NodeRef<marker::Mut<'a>, K, V, marker::LeafOrInternal>, marker::KV> {
    fn fix_left_border_of_left_edge(mut self) {
        while let Internal(internal_kv) = self.force() {
            self = internal_kv.fix_left_child(Left(())).into_node().first_kv();
            debug_assert!(self.reborrow().into_node().len() > MIN_LEN);
        }
    }

    fn fix_right_border_of_right_edge(mut self) {
        while let Internal(internal_kv) = self.force() {
            self = internal_kv.fix_right_child(Right(())).into_node().last_kv();
            debug_assert!(self.reborrow().into_node().len() > MIN_LEN);
        }
    }

    fn fix_left_border_of_right_edge(mut self) {
        while let Internal(internal_kv) = self.force() {
            let child_left_edge = internal_kv.fix_right_child(Left(()));
            self = match child_left_edge.right_kv() {
                Ok(child_right_kv) => {
                    child_right_kv.fix_left_border_of_left_edge();
                    return;
                }
                Err(child_left_edge) => child_left_edge.into_node().last_kv(),
            };
            debug_assert!(self.reborrow().into_node().len() > MIN_LEN);
        }
    }

    fn fix_right_border_of_left_edge(mut self) {
        while let Internal(internal_kv) = self.force() {
            let child_right_edge = internal_kv.fix_left_child(Right(()));
            self = match child_right_edge.left_kv() {
                Ok(child_left_kv) => {
                    child_left_kv.fix_right_border_of_right_edge();
                    return;
                }
                Err(child_right_edge) => child_right_edge.into_node().first_kv(),
            };
            debug_assert!(self.reborrow().into_node().len() > MIN_LEN);
        }
    }
}

impl<'a, K: 'a, V: 'a> Handle<NodeRef<marker::Mut<'a>, K, V, marker::Internal>, marker::Edge> {
    /// Fixes the child only, using one of the adjacent parent KVs.
    /// Returns the edge where the left child of the original child ended up.
    fn try_fixing_child_tracking_border(
        self,
        track_border: LeftOrRight<()>,
    ) -> Result<Handle<NodeRef<marker::Mut<'a>, K, V, marker::LeafOrInternal>, marker::Edge>, Self>
    {
        match self.left_kv() {
            Ok(left_kv) => Ok(left_kv.fix_right_child(track_border)),
            Err(edge) => match edge.right_kv() {
                Ok(right_kv) => Ok(right_kv.fix_left_child(track_border)),
                Err(edge) => Err(edge),
            },
        }
    }
}

impl<'a, K: 'a, V: 'a> Handle<NodeRef<marker::Mut<'a>, K, V, marker::Internal>, marker::KV> {
    /// Stocks up the left child, assuming the right child isn't underfull, and
    /// provisions an extra element to allow merging its children in turn
    /// without becoming underfull.
    /// Returns the edge where the tracked border of the original left child ended up.
    fn fix_left_child(
        self,
        track_border: LeftOrRight<()>,
    ) -> Handle<NodeRef<marker::Mut<'a>, K, V, marker::LeafOrInternal>, marker::Edge> {
        let mut internal_kv = self.consider_for_balancing();
        let left_len = internal_kv.left_child_len();
        debug_assert!(internal_kv.right_child_len() >= MIN_LEN);
        let track_edge_idx = if matches!(track_border, Left(_)) { 0 } else { left_len };
        if internal_kv.can_merge() {
            internal_kv.merge_tracking_child_edge(Left(track_edge_idx))
        } else {
            // `MIN_LEN + 1` to avoid readjust if merge happens on the next level.
            let count = (MIN_LEN + 1).saturating_sub(left_len);
            if count > 0 {
                internal_kv.bulk_steal_right(count);
            }
            unsafe { Handle::new_edge(internal_kv.into_left_child(), track_edge_idx) }
        }
    }

    /// Stocks up the right child, assuming the left child isn't underfull, and
    /// provisions an extra element to allow merging its children in turn
    /// without becoming underfull.
    /// Returns the edge where the tracked border of the original right child ended up.
    fn fix_right_child(
        self,
        track_border: LeftOrRight<()>,
    ) -> Handle<NodeRef<marker::Mut<'a>, K, V, marker::LeafOrInternal>, marker::Edge> {
        let mut internal_kv = self.consider_for_balancing();
        let right_len = internal_kv.right_child_len();
        debug_assert!(internal_kv.left_child_len() >= MIN_LEN);
        let track_edge_idx = if matches!(track_border, Left(_)) { 0 } else { right_len };
        if internal_kv.can_merge() {
            internal_kv.merge_tracking_child_edge(Right(track_edge_idx))
        } else {
            // `MIN_LEN + 1` to avoid readjust if merge happens on the next level.
            let count = (MIN_LEN + 1).saturating_sub(right_len);
            if count > 0 {
                internal_kv.bulk_steal_left(count);
            }
            unsafe { Handle::new_edge(internal_kv.into_right_child(), count + track_edge_idx) }
        }
    }
}
