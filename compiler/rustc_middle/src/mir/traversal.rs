use super::*;

/// Preorder traversal of a graph.
///
/// Preorder traversal is when each node is visited after at least one of its predecessors. If you
/// are familiar with some basic graph theory, then this performs a depth first search and returns
/// nodes in order of discovery time.
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
    visited: DenseBitSet<BasicBlock>,
    worklist: Vec<BasicBlock>,
}

impl<'a, 'tcx> Preorder<'a, 'tcx> {
    pub fn new(body: &'a Body<'tcx>, root: BasicBlock) -> Preorder<'a, 'tcx> {
        let worklist = vec![root];

        Preorder { body, visited: DenseBitSet::new_empty(body.basic_blocks.len()), worklist }
    }
}

/// Preorder traversal of a graph.
///
/// This function creates an iterator over the `Body`'s basic blocks, that
/// returns basic blocks in a preorder.
///
/// See [`Preorder`]'s docs to learn what is preorder traversal.
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
        // The worklist might be only things already visited.
        let lower = 0;

        // This is extremely loose, but it's not worth a popcnt loop to do better.
        let upper = self.body.basic_blocks.len();

        (lower, Some(upper))
    }
}

/// Postorder traversal of a graph.
///
/// Postorder traversal is when each node is visited after all of its successors, except when the
/// successor is only reachable by a back-edge. If you are familiar with some basic graph theory,
/// then this performs a depth first search and returns nodes in order of completion time.
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
    basic_blocks: &'a IndexSlice<BasicBlock, BasicBlockData<'tcx>>,
    visited: DenseBitSet<BasicBlock>,
    visit_stack: Vec<(BasicBlock, Successors<'a>)>,
    /// A non-empty `extra` allows for a precise calculation of the successors.
    extra: Option<(TyCtxt<'tcx>, Instance<'tcx>)>,
}

impl<'a, 'tcx> Postorder<'a, 'tcx> {
    pub fn new(
        basic_blocks: &'a IndexSlice<BasicBlock, BasicBlockData<'tcx>>,
        root: BasicBlock,
        extra: Option<(TyCtxt<'tcx>, Instance<'tcx>)>,
    ) -> Postorder<'a, 'tcx> {
        let mut po = Postorder {
            basic_blocks,
            visited: DenseBitSet::new_empty(basic_blocks.len()),
            visit_stack: Vec::new(),
            extra,
        };

        po.visit(root);
        po.traverse_successor();

        po
    }

    fn visit(&mut self, bb: BasicBlock) {
        if !self.visited.insert(bb) {
            return;
        }
        let data = &self.basic_blocks[bb];
        let successors = if let Some(extra) = self.extra {
            data.mono_successors(extra.0, extra.1)
        } else {
            data.terminator().successors()
        };
        self.visit_stack.push((bb, successors));
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
        //      |     D
        //       \   /
        //        \ /
        //         E
        //
        // The state of the stack starts out with just the root node (`A` in this case);
        //     [(A, [B, C])]
        //
        // When the first call to `traverse_successor` happens, the following happens:
        //
        //     [(C, [D]),  // `C` taken from the successors of `A`, pushed to the
        //                 // top of the stack along with the successors of `C`
        //      (A, [B])]
        //
        //     [(D, [E]),  // `D` taken from successors of `C`, pushed to stack
        //      (C, []),
        //      (A, [B])]
        //
        //     [(E, []),   // `E` taken from successors of `D`, pushed to stack
        //      (D, []),
        //      (C, []),
        //      (A, [B])]
        //
        // Now that the top of the stack has no successors we can traverse, each item will
        // be popped off during iteration until we get back to `A`. This yields [E, D, C].
        //
        // When we yield `C` and call `traverse_successor`, we push `B` to the stack, but
        // since we've already visited `E`, that child isn't added to the stack. The last
        // two iterations yield `B` and finally `A` for a final traversal of [E, D, C, B, A]
        while let Some(bb) = self.visit_stack.last_mut().and_then(|(_, iter)| iter.next_back()) {
            self.visit(bb);
        }
    }
}

impl<'tcx> Iterator for Postorder<'_, 'tcx> {
    type Item = BasicBlock;

    fn next(&mut self) -> Option<BasicBlock> {
        let (bb, _) = self.visit_stack.pop()?;
        self.traverse_successor();

        Some(bb)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        // These bounds are not at all tight, but that's fine.
        // It's not worth a popcnt loop in `DenseBitSet` to improve the upper,
        // and in mono-reachable we can't be precise anyway.
        // Leaning on amortized growth is fine.

        let lower = self.visit_stack.len();
        let upper = self.basic_blocks.len();
        (lower, Some(upper))
    }
}

/// Postorder traversal of a graph.
///
/// This function creates an iterator over the `Body`'s basic blocks, that:
/// - returns basic blocks in a postorder,
/// - traverses the `BasicBlocks` CFG cache's reverse postorder backwards, and does not cache the
///   postorder itself.
///
/// See [`Postorder`]'s docs to learn what is postorder traversal.
pub fn postorder<'a, 'tcx>(
    body: &'a Body<'tcx>,
) -> impl Iterator<Item = (BasicBlock, &'a BasicBlockData<'tcx>)> + ExactSizeIterator + DoubleEndedIterator
{
    reverse_postorder(body).rev()
}

pub fn mono_reachable_reverse_postorder<'a, 'tcx>(
    body: &'a Body<'tcx>,
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
) -> Vec<BasicBlock> {
    let mut iter = Postorder::new(&body.basic_blocks, START_BLOCK, Some((tcx, instance)));
    let mut items = Vec::with_capacity(body.basic_blocks.len());
    while let Some(block) = iter.next() {
        items.push(block);
    }
    items.reverse();
    items
}

/// Returns an iterator over all basic blocks reachable from the `START_BLOCK` in no particular
/// order.
///
/// This is clearer than writing `preorder` in cases where the order doesn't matter.
pub fn reachable<'a, 'tcx>(
    body: &'a Body<'tcx>,
) -> impl 'a + Iterator<Item = (BasicBlock, &'a BasicBlockData<'tcx>)> {
    preorder(body)
}

/// Returns a `DenseBitSet` containing all basic blocks reachable from the `START_BLOCK`.
pub fn reachable_as_bitset(body: &Body<'_>) -> DenseBitSet<BasicBlock> {
    let mut iter = preorder(body);
    while let Some(_) = iter.next() {}
    iter.visited
}

/// Reverse postorder traversal of a graph.
///
/// This function creates an iterator over the `Body`'s basic blocks, that:
/// - returns basic blocks in a reverse postorder,
/// - makes use of the `BasicBlocks` CFG cache's reverse postorder.
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
pub fn reverse_postorder<'a, 'tcx>(
    body: &'a Body<'tcx>,
) -> impl Iterator<Item = (BasicBlock, &'a BasicBlockData<'tcx>)> + ExactSizeIterator + DoubleEndedIterator
{
    body.basic_blocks.reverse_postorder().iter().map(|&bb| (bb, &body.basic_blocks[bb]))
}

/// Traversal of a [`Body`] that tries to avoid unreachable blocks in a monomorphized [`Instance`].
///
/// This is allowed to have false positives; blocks may be visited even if they are not actually
/// reachable.
///
/// Such a traversal is mostly useful because it lets us skip lowering the `false` side
/// of `if <T as Trait>::CONST`, as well as [`NullOp::UbChecks`].
///
/// [`NullOp::UbChecks`]: rustc_middle::mir::NullOp::UbChecks
pub fn mono_reachable<'a, 'tcx>(
    body: &'a Body<'tcx>,
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
) -> MonoReachable<'a, 'tcx> {
    MonoReachable::new(body, tcx, instance)
}

/// [`MonoReachable`] internally accumulates a [`DenseBitSet`] of visited blocks. This is just a
/// convenience function to run that traversal then extract its set of reached blocks.
pub fn mono_reachable_as_bitset<'a, 'tcx>(
    body: &'a Body<'tcx>,
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
) -> DenseBitSet<BasicBlock> {
    let mut iter = mono_reachable(body, tcx, instance);
    while let Some(_) = iter.next() {}
    iter.visited
}

pub struct MonoReachable<'a, 'tcx> {
    body: &'a Body<'tcx>,
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
    visited: DenseBitSet<BasicBlock>,
    // Other traversers track their worklist in a Vec. But we don't care about order, so we can
    // store ours in a DenseBitSet and thus save allocations because DenseBitSet has a small size
    // optimization.
    worklist: DenseBitSet<BasicBlock>,
}

impl<'a, 'tcx> MonoReachable<'a, 'tcx> {
    pub fn new(
        body: &'a Body<'tcx>,
        tcx: TyCtxt<'tcx>,
        instance: Instance<'tcx>,
    ) -> MonoReachable<'a, 'tcx> {
        let mut worklist = DenseBitSet::new_empty(body.basic_blocks.len());
        worklist.insert(START_BLOCK);
        MonoReachable {
            body,
            tcx,
            instance,
            visited: DenseBitSet::new_empty(body.basic_blocks.len()),
            worklist,
        }
    }

    fn add_work(&mut self, blocks: impl IntoIterator<Item = BasicBlock>) {
        for block in blocks.into_iter() {
            if !self.visited.contains(block) {
                self.worklist.insert(block);
            }
        }
    }
}

impl<'a, 'tcx> Iterator for MonoReachable<'a, 'tcx> {
    type Item = (BasicBlock, &'a BasicBlockData<'tcx>);

    fn next(&mut self) -> Option<(BasicBlock, &'a BasicBlockData<'tcx>)> {
        while let Some(idx) = self.worklist.iter().next() {
            self.worklist.remove(idx);
            if !self.visited.insert(idx) {
                continue;
            }

            let data = &self.body[idx];

            let targets = data.mono_successors(self.tcx, self.instance);
            self.add_work(targets);

            return Some((idx, data));
        }

        None
    }
}
