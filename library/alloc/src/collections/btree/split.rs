use super::node::LeftOrRight::{self, *};
use super::node::{marker, ForceResult::*, Handle, NodeRef, Root};
use super::search::{SearchBound, SearchResult::*};
use core::borrow::Borrow;
use core::ops::RangeBounds;
use core::ptr;

impl<K, V> Root<K, V> {
    /// Calculates the length of both trees that result from splitting up
    /// a given number of distinct key-value pairs.
    pub fn calc_split_length(
        total_num: usize,
        root_a: &Root<K, V>,
        root_b: &Root<K, V>,
    ) -> (usize, usize) {
        let (length_a, length_b);
        if root_a.height() < root_b.height() {
            length_a = root_a.reborrow().calc_length();
            length_b = total_num - length_a;
            debug_assert_eq!(length_b, root_b.reborrow().calc_length());
        } else {
            length_b = root_b.reborrow().calc_length();
            length_a = total_num - length_b;
            debug_assert_eq!(length_a, root_a.reborrow().calc_length());
        }
        (length_a, length_b)
    }

    /// Split off a tree with key-value pairs at and after the given key.
    /// The result is meaningful only if the tree is ordered by key,
    /// and if the ordering of `Q` corresponds to that of `K`.
    /// If `self` respects all `BTreeMap` tree invariants, then both
    /// `self` and the returned tree will respect those invariants.
    pub fn split_off<Q: ?Sized + Ord>(&mut self, key: &Q) -> Self
    where
        K: Borrow<Q>,
    {
        let left_root = self;
        let mut right_root = Root::new_pillar(left_root.height());
        let mut left_node = left_root.borrow_mut();
        let mut right_node = right_root.borrow_mut();

        loop {
            let mut split_edge = match left_node.search_node(key) {
                // key is going to the right tree
                Found(kv) => kv.left_edge(),
                GoDown(edge) => edge,
            };

            split_edge.move_suffix(&mut right_node);

            match (split_edge.force(), right_node.force()) {
                (Internal(edge), Internal(node)) => {
                    left_node = edge.descend();
                    right_node = node.first_edge().descend();
                }
                (Leaf(_), Leaf(_)) => break,
                _ => unreachable!(),
            }
        }

        left_root.fix_right_border();
        right_root.fix_left_border();
        right_root
    }

    /// Splits off a tree with the key-value pairs contained within a range.
    /// The returned tree respects the tree invariants of the `node` module,
    /// and has keys in ascending order, but does not care about the other
    /// tree invariants of `BTreeMap`. Invoke its `fix_both_borders` to make
    /// it a valid root for a `BTreeMap`.
    pub fn split_off_range<Q, R>(&mut self, range: R) -> Self
    where
        Q: ?Sized + Ord,
        K: Borrow<Q>,
        R: RangeBounds<Q>,
    {
        let (node, lower_edge_idx, upper_edge_idx, lower_child_bound, upper_child_bound) =
            match self.borrow_mut().search_tree_for_bifurcation(&range) {
                Ok(found) => found,
                Err(_) => return Root::new(),
            };
        debug_assert!(lower_edge_idx < upper_edge_idx);

        // We're at the highest node that contains a non-empty sequence of
        // key-value pairs within the range. Move out those pairs, and any edge
        // in between them, to form the root of the new tree.
        // Then descend further along the left and right border, on each level
        // splitting up one node along the lower bound (start of the range), and
        // splitting up one node along the upper bound (end of the range),
        // resulting in:
        // - Lower stems, the parts remaining along the lower bound.
        // - Lower clippings, the parts split off along the lower bound.
        // - Upper stems, the parts remaining along the upper bound, that belong
        //   in the new tree.
        // - Upper clippings, the parts split off along the upper bound, that
        //   belong in `self`.
        // Each of those can easily be underfull or empty, which makes it hard
        // to preserve the `BTreeMap` invariants.
        //
        // In addition to that, repairing `self` is hard because we're left with
        // with two edges and no key-value pair to glue them together. Therefore,
        // we dig up a key-value pair from one of the edges.
        //
        // Constructing the new tree is merely gluing two outer edges to the
        // moved sequence of key-value pairs.
        match node.force() {
            Leaf(leaf) => {
                let mut leaf_edge = unsafe { Handle::new_edge(leaf, lower_edge_idx) };
                let mut new_root = NodeRef::new_leaf();
                leaf_edge.move_infix(upper_edge_idx, &mut new_root.borrow_mut());
                leaf_edge.into_node().forget_type().fix_node_and_affected_ancestors();
                self.fix_top();
                new_root.forget_type()
            }
            Internal(node) => {
                let child_height = node.height() - 1;
                let mut lower_edge = unsafe { Handle::new_edge(ptr::read(&node), lower_edge_idx) };
                let upper_edge = unsafe { Handle::new_edge(node, upper_edge_idx) };
                let lower_child = unsafe { ptr::read(&lower_edge) }.descend();
                let upper_child = unsafe { ptr::read(&upper_edge) }.descend();
                let mut lower_clippings = Root::new_pillar(child_height);
                let mut upper_clippings = Root::new_pillar(child_height);
                let middle_kv = lower_child.split_off_bound(
                    lower_clippings.borrow_mut(),
                    lower_child_bound,
                    Left(()),
                );
                upper_child.split_off_bound(
                    upper_clippings.borrow_mut(),
                    upper_child_bound,
                    Right(()),
                );
                // We keep the edge left of the infix (minus the part split off into
                // `upper_clippings`), move out the infix and the edge right of
                // the infix (minus the part split off), and replace it with
                // `upper_clippings`. But we need a KV to bridge the edges.
                let mut dst_root = NodeRef::new_internal(lower_clippings);
                if let Some(middle_kv) = middle_kv {
                    let mut lower_kv = lower_edge.right_kv().ok().unwrap();
                    let first_kv = lower_kv.replace_kv(middle_kv.0, middle_kv.1);
                    let mut lower_edge_plus_1 = unsafe { lower_kv.reborrow_mut() }.right_edge();
                    let upper_stems = lower_edge_plus_1.replace_edge(upper_clippings);
                    dst_root.borrow_mut().push(first_kv.0, first_kv.1, upper_stems);
                    lower_edge_plus_1.move_infix(upper_edge.idx(), &mut dst_root.borrow_mut());
                    if lower_kv.forget_node_type().try_fixing_opposite_borders_and_ancestors() {
                        self.fix_top();
                    } else {
                        self.fix_opposite_borders();
                    }
                } else {
                    // There's no KV in the remainder of the left border, so instead we
                    // discard that empty remainder.
                    let lower_stems =
                        unsafe { lower_edge.reborrow_mut() }.replace_edge(upper_clippings);
                    lower_stems.deallocate_pillar();
                    lower_edge.move_infix(upper_edge.idx(), &mut dst_root.borrow_mut());
                    if lower_edge.try_fixing_left_border_and_ancestors() {
                        self.fix_top();
                    } else {
                        self.fix_left_border();
                    }
                }
                dst_root.forget_type()
            }
        }
    }

    /// Creates a tree consisting of empty nodes.
    fn new_pillar(height: usize) -> Self {
        let mut root = Root::new();
        for _ in 0..height {
            root.push_internal_level();
        }
        root
    }

    /// Destroys a tree consisting of empty nodes.
    fn deallocate_pillar(mut self) {
        debug_assert!(self.reborrow().calc_length() == 0);
        while self.height() > 0 {
            self.pop_internal_level();
        }
        unsafe { self.into_dying().deallocate_and_ascend() };
    }
}

impl<'a, K: 'a, V: 'a> NodeRef<marker::Mut<'a>, K, V, marker::LeafOrInternal> {
    /// Splits off a subtree at a given left (lower) or right (upper) bound.
    /// For the left bound, also splits off the greatest key-value pair below
    /// the bound, if any, and returns it.
    fn split_off_bound<'q, Q>(
        mut self,
        mut dst_node: Self,
        mut bound: SearchBound<&'q Q>,
        border: LeftOrRight<()>,
    ) -> Option<(K, V)>
    where
        Q: ?Sized + Ord,
        K: Borrow<Q>,
    {
        let top_height = self.height();
        loop {
            let (mut next_edge, next_bound) = match border {
                Left(_) => self.find_lower_bound_edge(bound),
                Right(_) => self.find_upper_bound_edge(bound),
            };
            // Possible optimization: if the entire node needs to be split off,
            // do not move the contents but swap the nodes instead.
            next_edge.move_suffix(&mut dst_node);
            match (next_edge.force(), dst_node.force()) {
                (Internal(next_edge), Internal(internal_dst_node)) => {
                    self = next_edge.descend();
                    dst_node = internal_dst_node.first_edge().descend();
                    bound = next_bound;
                }
                (Leaf(next_edge), Leaf(_)) => {
                    return match border {
                        Left(_) => next_edge.into_node().pop_last_kv(top_height),
                        Right(_) => None,
                    };
                }
                _ => unreachable!(),
            }
        }
    }
}

impl<'a, K: 'a, V: 'a> NodeRef<marker::Mut<'a>, K, V, marker::Leaf> {
    /// Backs up from a leaf node to the last KV within a subtree, and pops it.
    fn pop_last_kv(self, up_to_height: usize) -> Option<(K, V)> {
        let mut node = self.forget_type();
        while node.len() == 0 {
            if node.height() == up_to_height {
                return None;
            }
            node = node.ascend().ok().unwrap().into_node().forget_type();
        }
        let (k, v, empty_edge) = node.pop();
        if let Some(empty_edge) = empty_edge {
            empty_edge.deallocate_pillar();
        }
        Some((k, v))
    }
}
