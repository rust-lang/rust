// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::vec;

use rustc_data_structures::bitvec::BitVector;
use rustc_data_structures::indexed_vec::Idx;

use super::*;

/// Preorder traversal of a graph.
///
/// Preorder traversal is when each node is visited before an of it's
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
pub struct Preorder<'a, 'tcx: 'a> {
    mir: &'a Mir<'tcx>,
    visited: BitVector,
    worklist: Vec<BasicBlock>,
}

impl<'a, 'tcx> Preorder<'a, 'tcx> {
    pub fn new(mir: &'a Mir<'tcx>, root: BasicBlock) -> Preorder<'a, 'tcx> {
        let worklist = vec![root];

        Preorder {
            mir: mir,
            visited: BitVector::new(mir.basic_blocks().len()),
            worklist: worklist
        }
    }
}

pub fn preorder<'a, 'tcx>(mir: &'a Mir<'tcx>) -> Preorder<'a, 'tcx> {
    Preorder::new(mir, START_BLOCK)
}

impl<'a, 'tcx> Iterator for Preorder<'a, 'tcx> {
    type Item = (BasicBlock, &'a BasicBlockData<'tcx>);

    fn next(&mut self) -> Option<(BasicBlock, &'a BasicBlockData<'tcx>)> {
        while let Some(idx) = self.worklist.pop() {
            if !self.visited.insert(idx.index()) {
                continue;
            }

            let data = &self.mir[idx];

            if let Some(ref term) = data.terminator {
                for &succ in term.successors().iter() {
                    self.worklist.push(succ);
                }
            }

            return Some((idx, data));
        }

        None
    }
}

/// Postorder traversal of a graph.
///
/// Postorder traversal is when each node is visited after all of it's
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
pub struct Postorder<'a, 'tcx: 'a> {
    mir: &'a Mir<'tcx>,
    visited: BitVector,
    visit_stack: Vec<(BasicBlock, vec::IntoIter<BasicBlock>)>
}

impl<'a, 'tcx> Postorder<'a, 'tcx> {
    pub fn new(mir: &'a Mir<'tcx>, root: BasicBlock) -> Postorder<'a, 'tcx> {
        let mut po = Postorder {
            mir: mir,
            visited: BitVector::new(mir.basic_blocks().len()),
            visit_stack: Vec::new()
        };


        let data = &po.mir[root];

        if let Some(ref term) = data.terminator {
            po.visited.insert(root.index());

            let succs = term.successors().into_owned().into_iter();

            po.visit_stack.push((root, succs));
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
        // iterators over the sucessors of those nodes. Each iteration attempts to get the next
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
        // When the first call to `traverse_sucessor` happens, the following happens:
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
        // be popped off during iteration until we get back to `A`. This yeilds [E, D, B].
        //
        // When we yield `B` and call `traverse_successor`, we push `C` to the stack, but
        // since we've already visited `E`, that child isn't added to the stack. The last
        // two iterations yield `C` and finally `A` for a final traversal of [E, D, B, C, A]
        loop {
            let bb = if let Some(&mut (_, ref mut iter)) = self.visit_stack.last_mut() {
                if let Some(bb) = iter.next() {
                    bb
                } else {
                    break;
                }
            } else {
                break;
            };

            if self.visited.insert(bb.index()) {
                if let Some(ref term) = self.mir[bb].terminator {
                    let succs = term.successors().into_owned().into_iter();
                    self.visit_stack.push((bb, succs));
                }
            }
        }
    }
}

pub fn postorder<'a, 'tcx>(mir: &'a Mir<'tcx>) -> Postorder<'a, 'tcx> {
    Postorder::new(mir, START_BLOCK)
}

impl<'a, 'tcx> Iterator for Postorder<'a, 'tcx> {
    type Item = (BasicBlock, &'a BasicBlockData<'tcx>);

    fn next(&mut self) -> Option<(BasicBlock, &'a BasicBlockData<'tcx>)> {
        let next = self.visit_stack.pop();
        if next.is_some() {
            self.traverse_successor();
        }

        next.map(|(bb, _)| (bb, &self.mir[bb]))
    }
}

/// Reverse postorder traversal of a graph
///
/// Reverse postorder is the reverse order of a postorder traversal.
/// This is different to a preorder traversal and represents a natural
/// linearisation of control-flow.
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
/// Note that for a graph containing no loops (i.e. A DAG), this is equivalent to
/// a topological sort.
///
/// Construction of a `ReversePostorder` traversal requires doing a full
/// postorder traversal of the graph, therefore this traversal should be
/// constructed as few times as possible. Use the `reset` method to be able
/// to re-use the traversal
#[derive(Clone)]
pub struct ReversePostorder<'a, 'tcx: 'a> {
    mir: &'a Mir<'tcx>,
    blocks: Vec<BasicBlock>,
    idx: usize
}

impl<'a, 'tcx> ReversePostorder<'a, 'tcx> {
    pub fn new(mir: &'a Mir<'tcx>, root: BasicBlock) -> ReversePostorder<'a, 'tcx> {
        let blocks : Vec<_> = Postorder::new(mir, root).map(|(bb, _)| bb).collect();

        let len = blocks.len();

        ReversePostorder {
            mir: mir,
            blocks: blocks,
            idx: len
        }
    }

    pub fn reset(&mut self) {
        self.idx = self.blocks.len();
    }
}


pub fn reverse_postorder<'a, 'tcx>(mir: &'a Mir<'tcx>) -> ReversePostorder<'a, 'tcx> {
    ReversePostorder::new(mir, START_BLOCK)
}

impl<'a, 'tcx> Iterator for ReversePostorder<'a, 'tcx> {
    type Item = (BasicBlock, &'a BasicBlockData<'tcx>);

    fn next(&mut self) -> Option<(BasicBlock, &'a BasicBlockData<'tcx>)> {
        if self.idx == 0 { return None; }
        self.idx -= 1;

        self.blocks.get(self.idx).map(|&bb| (bb, &self.mir[bb]))
    }
}
