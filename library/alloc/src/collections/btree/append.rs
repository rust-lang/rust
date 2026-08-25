use core::alloc::Allocator;

use super::node::{self, Root};

impl<K, V> Root<K, V> {
    /// Pushes all key-value pairs to the end of the tree, incrementing a
    /// `length` variable along the way. The latter makes it easier for the
    /// caller to avoid a leak when the iterator panicks.
    pub(super) fn bulk_push<I, A: Allocator + Clone>(
        &mut self,
        iter: I,
        length: &mut usize,
        alloc: A,
    ) where
        I: Iterator<Item = (K, V)>,
    {
        let mut cur_node = self.borrow_mut().last_leaf_edge().into_node();
        // Iterate through all key-value pairs, pushing them into nodes at the right level.
        for (key, value) in iter {
            // Try to push key-value pair into the current leaf node.
            if cur_node.len() < node::CAPACITY {
                cur_node.push(key, value);
            } else {
                // No space left, go up and push there.
                let mut open_node;
                let mut test_node = cur_node.forget_type();
                loop {
                    match test_node.ascend() {
                        Ok(parent) => {
                            let parent = parent.into_node();
                            if parent.len() < node::CAPACITY {
                                // Found a node with space left, push here.
                                open_node = parent;
                                break;
                            } else {
                                // Go up again.
                                test_node = parent.forget_type();
                            }
                        }
                        Err(_) => {
                            // We are at the top, create a new root node and push there.
                            open_node = self.push_internal_level(alloc.clone());
                            break;
                        }
                    }
                }

                // Push key-value pair and new right subtree.
                let tree_height = open_node.height() - 1;
                let mut right_tree = Root::new(alloc.clone());
                for _ in 0..tree_height {
                    right_tree.push_internal_level(alloc.clone());
                }
                open_node.push(key, value, right_tree);

                // Go down to the rightmost leaf again.
                cur_node = open_node.forget_type().last_leaf_edge().into_node();
            }

            // Increment length every iteration, to make sure the map drops
            // the appended elements even if advancing the iterator panicks.
            *length += 1;
        }
        self.fix_right_border_of_plentiful();
    }
}
