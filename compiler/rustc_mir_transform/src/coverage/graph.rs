use std::cmp::Ordering;
use std::ops::{Index, IndexMut};
use std::{mem, slice};

use rustc_data_structures::fx::FxHashSet;
use rustc_data_structures::graph::dominators::Dominators;
use rustc_data_structures::graph::{self, DirectedGraph, StartNode};
use rustc_index::IndexVec;
use rustc_index::bit_set::DenseBitSet;
pub(crate) use rustc_middle::mir::coverage::{BasicCoverageBlock, START_BCB};
use rustc_middle::mir::{self, BasicBlock, Terminator, TerminatorKind};
use tracing::debug;

/// A coverage-specific simplification of the MIR control flow graph (CFG). The `CoverageGraph`s
/// nodes are `BasicCoverageBlock`s, which encompass one or more MIR `BasicBlock`s.
#[derive(Debug)]
pub(crate) struct CoverageGraph {
    bcbs: IndexVec<BasicCoverageBlock, BasicCoverageBlockData>,
    bb_to_bcb: IndexVec<BasicBlock, Option<BasicCoverageBlock>>,
    pub(crate) successors: IndexVec<BasicCoverageBlock, Vec<BasicCoverageBlock>>,
    pub(crate) predecessors: IndexVec<BasicCoverageBlock, Vec<BasicCoverageBlock>>,

    dominators: Option<Dominators<BasicCoverageBlock>>,
    /// Allows nodes to be compared in some total order such that _if_
    /// `a` dominates `b`, then `a < b`. If neither node dominates the other,
    /// their relative order is consistent but arbitrary.
    dominator_order_rank: IndexVec<BasicCoverageBlock, u32>,
    /// A loop header is a node that dominates one or more of its predecessors.
    is_loop_header: DenseBitSet<BasicCoverageBlock>,
    /// For each node, the loop header node of its nearest enclosing loop.
    /// This forms a linked list that can be traversed to find all enclosing loops.
    enclosing_loop_header: IndexVec<BasicCoverageBlock, Option<BasicCoverageBlock>>,
}

impl CoverageGraph {
    pub(crate) fn from_mir(mir_body: &mir::Body<'_>) -> Self {
        let (bcbs, bb_to_bcb) = Self::compute_basic_coverage_blocks(mir_body);

        // Pre-transform MIR `BasicBlock` successors and predecessors into the BasicCoverageBlock
        // equivalents. Note that since the BasicCoverageBlock graph has been fully simplified, the
        // each predecessor of a BCB leader_bb should be in a unique BCB. It is possible for a
        // `SwitchInt` to have multiple targets to the same destination `BasicBlock`, so
        // de-duplication is required. This is done without reordering the successors.

        let successors = IndexVec::<BasicCoverageBlock, _>::from_fn_n(
            |bcb| {
                let mut seen_bcbs = FxHashSet::default();
                let terminator = mir_body[bcbs[bcb].last_bb()].terminator();
                bcb_filtered_successors(terminator)
                    .into_iter()
                    .filter_map(|successor_bb| bb_to_bcb[successor_bb])
                    // Remove duplicate successor BCBs, keeping only the first.
                    .filter(|&successor_bcb| seen_bcbs.insert(successor_bcb))
                    .collect::<Vec<_>>()
            },
            bcbs.len(),
        );

        let mut predecessors = IndexVec::from_elem(Vec::new(), &bcbs);
        for (bcb, bcb_successors) in successors.iter_enumerated() {
            for &successor in bcb_successors {
                predecessors[successor].push(bcb);
            }
        }

        let num_nodes = bcbs.len();
        let mut this = Self {
            bcbs,
            bb_to_bcb,
            successors,
            predecessors,
            dominators: None,
            dominator_order_rank: IndexVec::from_elem_n(0, num_nodes),
            is_loop_header: DenseBitSet::new_empty(num_nodes),
            enclosing_loop_header: IndexVec::from_elem_n(None, num_nodes),
        };
        assert_eq!(num_nodes, this.num_nodes());

        // Set the dominators first, because later init steps rely on them.
        this.dominators = Some(graph::dominators::dominators(&this));

        // Iterate over all nodes, such that dominating nodes are visited before
        // the nodes they dominate. Either preorder or reverse postorder is fine.
        let dominator_order = graph::iterate::reverse_post_order(&this, this.start_node());
        // The coverage graph is created by traversal, so all nodes are reachable.
        assert_eq!(dominator_order.len(), this.num_nodes());
        for (rank, bcb) in (0u32..).zip(dominator_order) {
            // The dominator rank of each node is its index in a dominator-order traversal.
            this.dominator_order_rank[bcb] = rank;

            // A node is a loop header if it dominates any of its predecessors.
            if this.reloop_predecessors(bcb).next().is_some() {
                this.is_loop_header.insert(bcb);
            }

            // If the immediate dominator is a loop header, that's our enclosing loop.
            // Otherwise, inherit the immediate dominator's enclosing loop.
            // (Dominator order ensures that we already processed the dominator.)
            if let Some(dom) = this.dominators().immediate_dominator(bcb) {
                this.enclosing_loop_header[bcb] = this
                    .is_loop_header
                    .contains(dom)
                    .then_some(dom)
                    .or_else(|| this.enclosing_loop_header[dom]);
            }
        }

        // The coverage graph's entry-point node (bcb0) always starts with bb0,
        // which never has predecessors. Any other blocks merged into bcb0 can't
        // have multiple (coverage-relevant) predecessors, so bcb0 always has
        // zero in-edges.
        assert!(this[START_BCB].leader_bb() == mir::START_BLOCK);
        assert!(this.predecessors[START_BCB].is_empty());

        this
    }

    fn compute_basic_coverage_blocks(
        mir_body: &mir::Body<'_>,
    ) -> (
        IndexVec<BasicCoverageBlock, BasicCoverageBlockData>,
        IndexVec<BasicBlock, Option<BasicCoverageBlock>>,
    ) {
        let num_basic_blocks = mir_body.basic_blocks.len();
        let mut bcbs = IndexVec::<BasicCoverageBlock, _>::with_capacity(num_basic_blocks);
        let mut bb_to_bcb = IndexVec::from_elem_n(None, num_basic_blocks);

        let mut flush_chain_into_new_bcb = |current_chain: &mut Vec<BasicBlock>| {
            // Take the accumulated list of blocks, leaving the vector empty
            // to be used by subsequent BCBs.
            let basic_blocks = mem::take(current_chain);

            let bcb = bcbs.next_index();
            for &bb in basic_blocks.iter() {
                bb_to_bcb[bb] = Some(bcb);
            }

            let is_out_summable = basic_blocks.last().is_some_and(|&bb| {
                bcb_filtered_successors(mir_body[bb].terminator()).is_out_summable()
            });
            let bcb_data = BasicCoverageBlockData { basic_blocks, is_out_summable };
            debug!("adding {bcb:?}: {bcb_data:?}");
            bcbs.push(bcb_data);
        };

        // Traverse the MIR control-flow graph, accumulating chains of blocks
        // that can be combined into a single node in the coverage graph.
        // A depth-first search ensures that if two nodes can be chained
        // together, they will be adjacent in the traversal order.

        // Accumulates a chain of blocks that will be combined into one BCB.
        let mut current_chain = vec![];

        let subgraph = CoverageRelevantSubgraph::new(&mir_body.basic_blocks);
        for bb in graph::depth_first_search(subgraph, mir::START_BLOCK)
            .filter(|&bb| mir_body[bb].terminator().kind != TerminatorKind::Unreachable)
        {
            if let Some(&prev) = current_chain.last() {
                // Adding a block to a non-empty chain is allowed if the
                // previous block permits chaining, and the current block has
                // `prev` as its sole predecessor.
                let can_chain = subgraph.coverage_successors(prev).is_out_chainable()
                    && mir_body.basic_blocks.predecessors()[bb].as_slice() == &[prev];
                if !can_chain {
                    // The current block can't be added to the existing chain, so
                    // flush that chain into a new BCB, and start a new chain.
                    flush_chain_into_new_bcb(&mut current_chain);
                }
            }

            current_chain.push(bb);
        }

        if !current_chain.is_empty() {
            debug!("flushing accumulated blocks into one last BCB");
            flush_chain_into_new_bcb(&mut current_chain);
        }

        (bcbs, bb_to_bcb)
    }

    #[inline(always)]
    pub(crate) fn iter_enumerated(
        &self,
    ) -> impl Iterator<Item = (BasicCoverageBlock, &BasicCoverageBlockData)> {
        self.bcbs.iter_enumerated()
    }

    #[inline(always)]
    pub(crate) fn bcb_from_bb(&self, bb: BasicBlock) -> Option<BasicCoverageBlock> {
        if bb.index() < self.bb_to_bcb.len() { self.bb_to_bcb[bb] } else { None }
    }

    #[inline(always)]
    fn dominators(&self) -> &Dominators<BasicCoverageBlock> {
        self.dominators.as_ref().unwrap()
    }

    #[inline(always)]
    pub(crate) fn dominates(&self, dom: BasicCoverageBlock, node: BasicCoverageBlock) -> bool {
        self.dominators().dominates(dom, node)
    }

    #[inline(always)]
    pub(crate) fn cmp_in_dominator_order(
        &self,
        a: BasicCoverageBlock,
        b: BasicCoverageBlock,
    ) -> Ordering {
        self.dominator_order_rank[a].cmp(&self.dominator_order_rank[b])
    }

    /// For the given node, yields the subset of its predecessor nodes that
    /// it dominates. If that subset is non-empty, the node is a "loop header",
    /// and each of those predecessors represents an in-edge that jumps back to
    /// the top of its loop.
    pub(crate) fn reloop_predecessors(
        &self,
        to_bcb: BasicCoverageBlock,
    ) -> impl Iterator<Item = BasicCoverageBlock> {
        self.predecessors[to_bcb].iter().copied().filter(move |&pred| self.dominates(to_bcb, pred))
    }
}

impl Index<BasicCoverageBlock> for CoverageGraph {
    type Output = BasicCoverageBlockData;

    #[inline]
    fn index(&self, index: BasicCoverageBlock) -> &BasicCoverageBlockData {
        &self.bcbs[index]
    }
}

impl IndexMut<BasicCoverageBlock> for CoverageGraph {
    #[inline]
    fn index_mut(&mut self, index: BasicCoverageBlock) -> &mut BasicCoverageBlockData {
        &mut self.bcbs[index]
    }
}

impl graph::DirectedGraph for CoverageGraph {
    type Node = BasicCoverageBlock;

    #[inline]
    fn num_nodes(&self) -> usize {
        self.bcbs.len()
    }
}

impl graph::StartNode for CoverageGraph {
    #[inline]
    fn start_node(&self) -> Self::Node {
        self.bcb_from_bb(mir::START_BLOCK)
            .expect("mir::START_BLOCK should be in a BasicCoverageBlock")
    }
}

impl graph::Successors for CoverageGraph {
    #[inline]
    fn successors(&self, node: Self::Node) -> impl Iterator<Item = Self::Node> {
        self.successors[node].iter().copied()
    }
}

impl graph::Predecessors for CoverageGraph {
    #[inline]
    fn predecessors(&self, node: Self::Node) -> impl Iterator<Item = Self::Node> {
        self.predecessors[node].iter().copied()
    }
}

/// `BasicCoverageBlockData` holds the data indexed by a `BasicCoverageBlock`.
///
/// A `BasicCoverageBlock` (BCB) represents the maximal-length sequence of MIR `BasicBlock`s without
/// conditional branches, and form a new, simplified, coverage-specific Control Flow Graph, without
/// altering the original MIR CFG.
///
/// Note that running the MIR `SimplifyCfg` transform is not sufficient (and therefore not
/// necessary). The BCB-based CFG is a more aggressive simplification. For example:
///
///   * The BCB CFG ignores (trims) branches not relevant to coverage, such as unwind-related code,
///     that is injected by the Rust compiler but has no physical source code to count. This also
///     means a BasicBlock with a `Call` terminator can be merged into its primary successor target
///     block, in the same BCB. (But, note: Issue #78544: "MIR InstrumentCoverage: Improve coverage
///     of `#[should_panic]` tests and `catch_unwind()` handlers")
///   * Some BasicBlock terminators support Rust-specific concerns--like borrow-checking--that are
///     not relevant to coverage analysis. `FalseUnwind`, for example, can be treated the same as
///     a `Goto`, and merged with its successor into the same BCB.
///
/// Each BCB with at least one computed coverage span will have no more than one `Counter`.
/// In some cases, a BCB's execution count can be computed by `Expression`. Additional
/// disjoint coverage spans in a BCB can also be counted by `Expression` (by adding `ZERO`
/// to the BCB's primary counter or expression).
///
/// The BCB CFG is critical to simplifying the coverage analysis by ensuring graph path-based
/// queries (`dominates()`, `predecessors`, `successors`, etc.) have branch (control flow)
/// significance.
#[derive(Debug, Clone)]
pub(crate) struct BasicCoverageBlockData {
    pub(crate) basic_blocks: Vec<BasicBlock>,

    /// If true, this node's execution count can be assumed to be the sum of the
    /// execution counts of all of its **out-edges** (assuming no panics).
    ///
    /// Notably, this is false for a node ending with [`TerminatorKind::Yield`],
    /// because the yielding coroutine might not be resumed.
    pub(crate) is_out_summable: bool,
}

impl BasicCoverageBlockData {
    #[inline(always)]
    pub(crate) fn leader_bb(&self) -> BasicBlock {
        self.basic_blocks[0]
    }

    #[inline(always)]
    pub(crate) fn last_bb(&self) -> BasicBlock {
        *self.basic_blocks.last().unwrap()
    }
}

/// Holds the coverage-relevant successors of a basic block's terminator, and
/// indicates whether that block can potentially be combined into the same BCB
/// as its sole successor.
#[derive(Clone, Copy, Debug)]
struct CoverageSuccessors<'a> {
    /// Coverage-relevant successors of the corresponding terminator.
    /// There might be 0, 1, or multiple targets.
    targets: &'a [BasicBlock],
    /// `Yield` terminators are not chainable, because their sole out-edge is
    /// only followed if/when the generator is resumed after the yield.
    is_yield: bool,
}

impl CoverageSuccessors<'_> {
    /// If `false`, this terminator cannot be chained into another block when
    /// building the coverage graph.
    fn is_out_chainable(&self) -> bool {
        // If a terminator is out-summable and has exactly one out-edge, then
        // it is eligible to be chained into its successor block.
        self.is_out_summable() && self.targets.len() == 1
    }

    /// Returns true if the terminator itself is assumed to have the same
    /// execution count as the sum of its out-edges (assuming no panics).
    fn is_out_summable(&self) -> bool {
        !self.is_yield && !self.targets.is_empty()
    }
}

impl IntoIterator for CoverageSuccessors<'_> {
    type Item = BasicBlock;
    type IntoIter = impl DoubleEndedIterator<Item = Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.targets.iter().copied()
    }
}

// Returns the subset of a block's successors that are relevant to the coverage
// graph, i.e. those that do not represent unwinds or false edges.
// FIXME(#78544): MIR InstrumentCoverage: Improve coverage of `#[should_panic]` tests and
// `catch_unwind()` handlers.
fn bcb_filtered_successors<'a, 'tcx>(terminator: &'a Terminator<'tcx>) -> CoverageSuccessors<'a> {
    use TerminatorKind::*;
    let mut is_yield = false;
    let targets = match &terminator.kind {
        // A switch terminator can have many coverage-relevant successors.
        SwitchInt { targets, .. } => targets.all_targets(),

        // A yield terminator has exactly 1 successor, but should not be chained,
        // because its resume edge has a different execution count.
        Yield { resume, .. } => {
            is_yield = true;
            slice::from_ref(resume)
        }

        // These terminators have exactly one coverage-relevant successor,
        // and can be chained into it.
        Assert { target, .. }
        | Drop { target, .. }
        | FalseEdge { real_target: target, .. }
        | FalseUnwind { real_target: target, .. }
        | Goto { target } => slice::from_ref(target),

        // A call terminator can normally be chained, except when it has no
        // successor because it is known to diverge.
        Call { target: maybe_target, .. } => maybe_target.as_slice(),

        // An inline asm terminator can normally be chained, except when it
        // diverges or uses asm goto.
        InlineAsm { targets, .. } => &targets,

        // These terminators have no coverage-relevant successors.
        CoroutineDrop
        | Return
        | TailCall { .. }
        | Unreachable
        | UnwindResume
        | UnwindTerminate(_) => &[],
    };

    CoverageSuccessors { targets, is_yield }
}

/// Wrapper around a [`mir::BasicBlocks`] graph that restricts each node's
/// successors to only the ones considered "relevant" when building a coverage
/// graph.
#[derive(Clone, Copy)]
struct CoverageRelevantSubgraph<'a, 'tcx> {
    basic_blocks: &'a mir::BasicBlocks<'tcx>,
}
impl<'a, 'tcx> CoverageRelevantSubgraph<'a, 'tcx> {
    fn new(basic_blocks: &'a mir::BasicBlocks<'tcx>) -> Self {
        Self { basic_blocks }
    }

    fn coverage_successors(&self, bb: BasicBlock) -> CoverageSuccessors<'_> {
        bcb_filtered_successors(self.basic_blocks[bb].terminator())
    }
}
impl<'a, 'tcx> graph::DirectedGraph for CoverageRelevantSubgraph<'a, 'tcx> {
    type Node = BasicBlock;

    fn num_nodes(&self) -> usize {
        self.basic_blocks.num_nodes()
    }
}
impl<'a, 'tcx> graph::Successors for CoverageRelevantSubgraph<'a, 'tcx> {
    fn successors(&self, bb: Self::Node) -> impl Iterator<Item = Self::Node> {
        self.coverage_successors(bb).into_iter()
    }
}
