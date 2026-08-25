use core::alloc::Allocator;
use core::borrow::Borrow;
use core::{intrinsics, mem};

use super::node::ForceResult::*;
use super::node::Root;
use super::search::SearchResult::*;

impl<K, V> Root<K, V> {
    /// Calculates the length of both trees that result from splitting up
    /// a given number of distinct key-value pairs.
    pub(super) fn calc_split_length(
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
    pub(super) fn split_off<Q: ?Sized + Ord, A: Allocator + Clone>(
        &mut self,
        key: &Q,
        alloc: A,
    ) -> Self
    where
        K: Borrow<Q>,
    {
        let left_root = self;
        let mut right_root = Root::new_pillar(left_root.height(), alloc.clone());
        let mut left_node = left_root.borrow_mut();
        let mut right_node = right_root.borrow_mut();

        // The first search runs before anything has moved, so a panic from the
        // caller's `Ord`/`Borrow` impl here can unwind safely: `self` is
        // untouched and the new right tree is still empty.
        let mut split_edge = match left_node.search_node(key) {
            // key is going to the right tree
            Found(kv) => kv.left_edge(),
            GoDown(edge) => edge,
        };

        // From the first `move_suffix` on, `left_root` and `right_root` share
        // key-value pairs through two temporarily invalid tree structures, and
        // neither is independently droppable until `fix_right_border` /
        // `fix_left_border` repair them and the caller recomputes both lengths.
        // A panic from a later `search_node` comparison would unwind out of
        // that state and double-free the shared values (#158165), so abort,
        // as `mem::replace` does elsewhere in this module.
        struct PanicGuard;
        impl Drop for PanicGuard {
            fn drop(&mut self) {
                intrinsics::abort()
            }
        }
        let guard = PanicGuard;

        loop {
            split_edge.move_suffix(&mut right_node);

            match (split_edge.force(), right_node.force()) {
                (Internal(edge), Internal(node)) => {
                    left_node = edge.descend();
                    right_node = node.first_edge().descend();
                }
                (Leaf(_), Leaf(_)) => break,
                _ => unreachable!(),
            }

            split_edge = match left_node.search_node(key) {
                Found(kv) => kv.left_edge(),
                GoDown(edge) => edge,
            };
        }

        left_root.fix_right_border(alloc.clone());
        right_root.fix_left_border(alloc);
        mem::forget(guard);
        right_root
    }

    /// Creates a tree consisting of empty nodes.
    fn new_pillar<A: Allocator + Clone>(height: usize, alloc: A) -> Self {
        let mut root = Root::new(alloc.clone());
        for _ in 0..height {
            root.push_internal_level(alloc.clone());
        }
        root
    }
}
