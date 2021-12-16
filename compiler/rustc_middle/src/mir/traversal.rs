use rustc_index::bit_set::BitSet;

use super::*;

/// Preorder traversal of a graph.
///
/// Preorder traversal is when each node is visited before any of its
/// successors
///
/// ```text
///
///         A
///        / \
///       /   \
///      B     C
///       \   /
///        \ /
///         D
/// ```
///
/// A preorder traversal of this graph is either `A B D C` or `A C D B`
#[derive(Clone)]
pub struct Preorder<'a, 'tcx> {
    body: &'a Body<'tcx>,
    visited: BitSet<BasicBlock>,
    worklist: Vec<BasicBlock>,
    root_is_start_block: bool,
}

impl<'a, 'tcx> Preorder<'a, 'tcx> {
    pub fn new(body: &'a Body<'tcx>, root: BasicBlock) -> Preorder<'a, 'tcx> {
        let worklist = vec![root];

        Preorder {
            body,
            visited: BitSet::new_empty(body.basic_blocks().len()),
            worklist,
            root_is_start_block: root == START_BLOCK,
        }
    }
}

pub fn preorder<'a, 'tcx>(body: &'a Body<'tcx>) -> Preorder<'a, 'tcx> {
    Preorder::new(body, START_BLOCK)
}

impl<'a, 'tcx> Iterator for Preorder<'a, 'tcx> {
    type Item = (BasicBlock, &'a BasicBlockData<'tcx>);

    fn next(&mut self) -> Option<(BasicBlock, &'a BasicBlockData<'tcx>)> {
        while let Some(idx) = self.worklist.pop() {
            if !self.visited.insert(idx) {
                continue;
            }

            let data = &self.body[idx];

            if let Some(ref term) = data.terminator {
                self.worklist.extend(term.successors());
            }

            return Some((idx, data));
        }

        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        // All the blocks, minus the number of blocks we've visited.
        let upper = self.body.basic_blocks().len() - self.visited.count();

        let lower = if self.root_is_start_block {
            // We will visit all remaining blocks exactly once.
            upper
        } else {
            self.worklist.len()
        };

        (lower, Some(upper))
    }
}

/// Postorder traversal of a graph.
///
/// Postorder traversal is when each node is visited after all of its
/// successors, except when the successor is only reachable by a back-edge
///
///
/// ```text
///
///         A
///        / \
///       /   \
///      B     C
///       \   /
///        \ /
///         D
/// ```
///
/// A Postorder traversal of this graph is `D B C A` or `D C B A`
pub struct Postorder<'a, 'tcx> {
    body: &'a Body<'tcx>,
    visited: BitSet<BasicBlock>,
    visit_stack: Vec<(BasicBlock, Successors<'a>)>,
    root_is_start_block: bool,
}

impl<'a, 'tcx> Postorder<'a, 'tcx> {
    pub fn new(body: &'a Body<'tcx>, root: BasicBlock) -> Postorder<'a, 'tcx> {
        let mut po = Postorder {
            body,
            visited: BitSet::new_empty(body.basic_blocks().len()),
            visit_stack: Vec::new(),
            root_is_start_block: root == START_BLOCK,
        };

        let data = &po.body[root];

        if let Some(ref term) = data.terminator {
            po.visited.insert(root);
            po.visit_stack.push((root, term.successors()));
            po.traverse_successor();
        }

        po
    }

    fn traverse_successor(&mut self) {
        // This is quite a complex loop due to 1. the borrow checker not liking it much
        // and 2. what exactly is going on is not clear
        //
        // It does the actual traversal of the graph, while the `next` method on the iterator
        // just pops off of the stack. `visit_stack` is a stack containing pairs of nodes and
        // iterators over the successors of those nodes. Each iteration attempts to get the next
        // node from the top of the stack, then pushes that node and an iterator over the
        // successors to the top of the stack. This loop only grows `visit_stack`, stopping when
        // we reach a child that has no children that we haven't already visited.
        //
        // For a graph that looks like this:
        //
        //         A
        //        / \
        //       /   \
        //      B     C
        //      |     |
        //      |     |
        //      D     |
        //       \   /
        //        \ /
        //         E
        //
        // The state of the stack starts out with just the root node (`A` in this case);
        //     [(A, [B, C])]
        //
        // When the first call to `traverse_successor` happens, the following happens:
        //
        //     [(B, [D]),  // `B` taken from the successors of `A`, pushed to the
        //                 // top of the stack along with the successors of `B`
        //      (A, [C])]
        //
        //     [(D, [E]),  // `D` taken from successors of `B`, pushed to stack
        //      (B, []),
        //      (A, [C])]
        //
        //     [(E, []),   // `E` taken from successors of `D`, pushed to stack
        //      (D, []),
        //      (B, []),
        //      (A, [C])]
        //
        // Now that the top of the stack has no successors we can traverse, each item will
        // be popped off during iteration until we get back to `A`. This yields [E, D, B].
        //
        // When we yield `B` and call `traverse_successor`, we push `C` to the stack, but
        // since we've already visited `E`, that child isn't added to the stack. The last
        // two iterations yield `C` and finally `A` for a final traversal of [E, D, B, C, A]
        loop {
            let bb = if let Some(&mut (_, ref mut iter)) = self.visit_stack.last_mut() {
                if let Some(&bb) = iter.next() {
                    bb
                } else {
                    break;
                }
            } else {
                break;
            };

            if self.visited.insert(bb) {
                if let Some(term) = &self.body[bb].terminator {
                    self.visit_stack.push((bb, term.successors()));
                }
            }
        }
    }
}

pub fn postorder<'a, 'tcx>(body: &'a Body<'tcx>) -> Postorder<'a, 'tcx> {
    Postorder::new(body, START_BLOCK)
}

impl<'a, 'tcx> Iterator for Postorder<'a, 'tcx> {
    type Item = (BasicBlock, &'a BasicBlockData<'tcx>);

    fn next(&mut self) -> Option<(BasicBlock, &'a BasicBlockData<'tcx>)> {
        let next = self.visit_stack.pop();
        if next.is_some() {
            self.traverse_successor();
        }

        next.map(|(bb, _)| (bb, &self.body[bb]))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        // All the blocks, minus the number of blocks we've visited.
        let upper = self.body.basic_blocks().len() - self.visited.count();

        let lower = if self.root_is_start_block {
            // We will visit all remaining blocks exactly once.
            upper
        } else {
            self.visit_stack.len()
        };

        (lower, Some(upper))
    }
}

/// Reverse postorder traversal of a graph
///
/// Reverse postorder is the reverse order of a postorder traversal.
/// This is different to a preorder traversal and represents a natural
/// linearization of control-flow.
///
/// ```text
///
///         A
///        / \
///       /   \
///      B     C
///       \   /
///        \ /
///         D
/// ```
///
/// A reverse postorder traversal of this graph is either `A B C D` or `A C B D`
/// Note that for a graph containing no loops (i.e., A DAG), this is equivalent to
/// a topological sort.
///
/// Construction of a `ReversePostorder` traversal requires doing a full
/// postorder traversal of the graph, therefore this traversal should be
/// constructed as few times as possible. Use the `reset` method to be able
/// to re-use the traversal
#[derive(Clone)]
pub struct ReversePostorder<'a, 'tcx> {
    body: &'a Body<'tcx>,
    blocks: Vec<BasicBlock>,
    idx: usize,
}

impl<'a, 'tcx> ReversePostorder<'a, 'tcx> {
    pub fn new(body: &'a Body<'tcx>, root: BasicBlock) -> ReversePostorder<'a, 'tcx> {
        let blocks: Vec<_> = Postorder::new(body, root).map(|(bb, _)| bb).collect();

        let len = blocks.len();

        ReversePostorder { body, blocks, idx: len }
    }
}

pub fn reverse_postorder<'a, 'tcx>(body: &'a Body<'tcx>) -> ReversePostorder<'a, 'tcx> {
    ReversePostorder::new(body, START_BLOCK)
}

impl<'a, 'tcx> Iterator for ReversePostorder<'a, 'tcx> {
    type Item = (BasicBlock, &'a BasicBlockData<'tcx>);

    fn next(&mut self) -> Option<(BasicBlock, &'a BasicBlockData<'tcx>)> {
        if self.idx == 0 {
            return None;
        }
        self.idx -= 1;

        self.blocks.get(self.idx).map(|&bb| (bb, &self.body[bb]))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.idx, Some(self.idx))
    }
}

impl<'a, 'tcx> ExactSizeIterator for ReversePostorder<'a, 'tcx> {}

/// Returns an iterator over all basic blocks reachable from the `START_BLOCK` in no particular
/// order.
///
/// This is clearer than writing `preorder` in cases where the order doesn't matter.
pub fn reachable<'a, 'tcx>(
    body: &'a Body<'tcx>,
) -> impl 'a + Iterator<Item = (BasicBlock, &'a BasicBlockData<'tcx>)> {
    preorder(body)
}

/// Returns a `BitSet` containing all basic blocks reachable from the `START_BLOCK`.
pub fn reachable_as_bitset<'tcx>(body: &Body<'tcx>) -> BitSet<BasicBlock> {
    let mut iter = preorder(body);
    (&mut iter).for_each(drop);
    iter.visited
}
