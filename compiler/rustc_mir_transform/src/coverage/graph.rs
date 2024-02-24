use rustc_data_structures::captures::Captures;
use rustc_data_structures::fx::FxHashSet;
use rustc_data_structures::graph::dominators::{self, Dominators};
use rustc_data_structures::graph::{self, GraphSuccessors, WithNumNodes, WithStartNode};
use rustc_index::bit_set::BitSet;
use rustc_index::IndexVec;
use rustc_middle::mir::{self, BasicBlock, Terminator, TerminatorKind};

use std::cmp::Ordering;
use std::collections::VecDeque;
use std::ops::{Index, IndexMut};

/// A coverage-specific simplification of the MIR control flow graph (CFG). The `CoverageGraph`s
/// nodes are `BasicCoverageBlock`s, which encompass one or more MIR `BasicBlock`s.
#[derive(Debug)]
pub(super) struct CoverageGraph {
    bcbs: IndexVec<BasicCoverageBlock, BasicCoverageBlockData>,
    bb_to_bcb: IndexVec<BasicBlock, Option<BasicCoverageBlock>>,
    pub successors: IndexVec<BasicCoverageBlock, Vec<BasicCoverageBlock>>,
    pub predecessors: IndexVec<BasicCoverageBlock, Vec<BasicCoverageBlock>>,
    dominators: Option<Dominators<BasicCoverageBlock>>,
}

impl CoverageGraph {
    pub fn from_mir(mir_body: &mir::Body<'_>) -> Self {
        let (bcbs, bb_to_bcb) = Self::compute_basic_coverage_blocks(mir_body);

        // Pre-transform MIR `BasicBlock` successors and predecessors into the BasicCoverageBlock
        // equivalents. Note that since the BasicCoverageBlock graph has been fully simplified, the
        // each predecessor of a BCB leader_bb should be in a unique BCB. It is possible for a
        // `SwitchInt` to have multiple targets to the same destination `BasicBlock`, so
        // de-duplication is required. This is done without reordering the successors.

        let successors = IndexVec::from_fn_n(
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

        let mut this = Self { bcbs, bb_to_bcb, successors, predecessors, dominators: None };

        this.dominators = Some(dominators::dominators(&this));

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

        let mut add_basic_coverage_block = |basic_blocks: &mut Vec<BasicBlock>| {
            // Take the accumulated list of blocks, leaving the vector empty
            // to be used by subsequent BCBs.
            let basic_blocks = std::mem::take(basic_blocks);

            let bcb = bcbs.next_index();
            for &bb in basic_blocks.iter() {
                bb_to_bcb[bb] = Some(bcb);
            }
            let bcb_data = BasicCoverageBlockData::from(basic_blocks);
            debug!("adding bcb{}: {:?}", bcb.index(), bcb_data);
            bcbs.push(bcb_data);
        };

        // Walk the MIR CFG using a Preorder traversal, which starts from `START_BLOCK` and follows
        // each block terminator's `successors()`. Coverage spans must map to actual source code,
        // so compiler generated blocks and paths can be ignored. To that end, the CFG traversal
        // intentionally omits unwind paths.
        // FIXME(#78544): MIR InstrumentCoverage: Improve coverage of `#[should_panic]` tests and
        // `catch_unwind()` handlers.

        // Accumulates a chain of blocks that will be combined into one BCB.
        let mut basic_blocks = Vec::new();

        let filtered_successors = |bb| bcb_filtered_successors(mir_body[bb].terminator());
        for bb in short_circuit_preorder(mir_body, filtered_successors)
            .filter(|&bb| mir_body[bb].terminator().kind != TerminatorKind::Unreachable)
        {
            // If the previous block can't be chained into `bb`, flush the accumulated
            // blocks into a new BCB, then start building the next chain.
            if let Some(&prev) = basic_blocks.last()
                && (!filtered_successors(prev).is_chainable() || {
                    // If `bb` has multiple predecessor blocks, or `prev` isn't
                    // one of its predecessors, we can't chain and must flush.
                    let predecessors = &mir_body.basic_blocks.predecessors()[bb];
                    predecessors.len() > 1 || !predecessors.contains(&prev)
                })
            {
                debug!(
                    terminator_kind = ?mir_body[prev].terminator().kind,
                    predecessors = ?&mir_body.basic_blocks.predecessors()[bb],
                    "can't chain from {prev:?} to {bb:?}"
                );
                add_basic_coverage_block(&mut basic_blocks);
            }

            basic_blocks.push(bb);
        }

        if !basic_blocks.is_empty() {
            debug!("flushing accumulated blocks into one last BCB");
            add_basic_coverage_block(&mut basic_blocks);
        }

        (bcbs, bb_to_bcb)
    }

    #[inline(always)]
    pub fn iter_enumerated(
        &self,
    ) -> impl Iterator<Item = (BasicCoverageBlock, &BasicCoverageBlockData)> {
        self.bcbs.iter_enumerated()
    }

    #[inline(always)]
    pub fn bcb_from_bb(&self, bb: BasicBlock) -> Option<BasicCoverageBlock> {
        if bb.index() < self.bb_to_bcb.len() { self.bb_to_bcb[bb] } else { None }
    }

    #[inline(always)]
    pub fn dominates(&self, dom: BasicCoverageBlock, node: BasicCoverageBlock) -> bool {
        self.dominators.as_ref().unwrap().dominates(dom, node)
    }

    #[inline(always)]
    pub fn cmp_in_dominator_order(&self, a: BasicCoverageBlock, b: BasicCoverageBlock) -> Ordering {
        self.dominators.as_ref().unwrap().cmp_in_dominator_order(a, b)
    }

    /// Returns true if the given node has 2 or more in-edges, i.e. 2 or more
    /// predecessors.
    ///
    /// This property is interesting to code that assigns counters to nodes and
    /// edges, because if a node _doesn't_ have multiple in-edges, then there's
    /// no benefit in having a separate counter for its in-edge, because it
    /// would have the same value as the node's own counter.
    ///
    /// FIXME: That assumption might not be true for [`TerminatorKind::Yield`]?
    #[inline(always)]
    pub(super) fn bcb_has_multiple_in_edges(&self, bcb: BasicCoverageBlock) -> bool {
        // Even though bcb0 conceptually has an extra virtual in-edge due to
        // being the entry point, we've already asserted that it has no _other_
        // in-edges, so there's no possibility of it having _multiple_ in-edges.
        // (And since its virtual in-edge doesn't exist in the graph, that edge
        // can't have a separate counter anyway.)
        self.predecessors[bcb].len() > 1
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
}

impl graph::WithNumNodes for CoverageGraph {
    #[inline]
    fn num_nodes(&self) -> usize {
        self.bcbs.len()
    }
}

impl graph::WithStartNode for CoverageGraph {
    #[inline]
    fn start_node(&self) -> Self::Node {
        self.bcb_from_bb(mir::START_BLOCK)
            .expect("mir::START_BLOCK should be in a BasicCoverageBlock")
    }
}

type BcbSuccessors<'graph> = std::slice::Iter<'graph, BasicCoverageBlock>;

impl<'graph> graph::GraphSuccessors<'graph> for CoverageGraph {
    type Item = BasicCoverageBlock;
    type Iter = std::iter::Cloned<BcbSuccessors<'graph>>;
}

impl graph::WithSuccessors for CoverageGraph {
    #[inline]
    fn successors(&self, node: Self::Node) -> <Self as GraphSuccessors<'_>>::Iter {
        self.successors[node].iter().cloned()
    }
}

impl<'graph> graph::GraphPredecessors<'graph> for CoverageGraph {
    type Item = BasicCoverageBlock;
    type Iter = std::iter::Copied<std::slice::Iter<'graph, BasicCoverageBlock>>;
}

impl graph::WithPredecessors for CoverageGraph {
    #[inline]
    fn predecessors(&self, node: Self::Node) -> <Self as graph::GraphPredecessors<'_>>::Iter {
        self.predecessors[node].iter().copied()
    }
}

rustc_index::newtype_index! {
    /// A node in the control-flow graph of CoverageGraph.
    #[orderable]
    #[debug_format = "bcb{}"]
    pub(super) struct BasicCoverageBlock {
        const START_BCB = 0;
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
pub(super) struct BasicCoverageBlockData {
    pub basic_blocks: Vec<BasicBlock>,
}

impl BasicCoverageBlockData {
    pub fn from(basic_blocks: Vec<BasicBlock>) -> Self {
        assert!(basic_blocks.len() > 0);
        Self { basic_blocks }
    }

    #[inline(always)]
    pub fn leader_bb(&self) -> BasicBlock {
        self.basic_blocks[0]
    }

    #[inline(always)]
    pub fn last_bb(&self) -> BasicBlock {
        *self.basic_blocks.last().unwrap()
    }
}

/// Holds the coverage-relevant successors of a basic block's terminator, and
/// indicates whether that block can potentially be combined into the same BCB
/// as its sole successor.
#[derive(Clone, Copy, Debug)]
enum CoverageSuccessors<'a> {
    /// The terminator has exactly one straight-line successor, so its block can
    /// potentially be combined into the same BCB as that successor.
    Chainable(BasicBlock),
    /// The block cannot be combined into the same BCB as its successor(s).
    NotChainable(&'a [BasicBlock]),
}

impl CoverageSuccessors<'_> {
    fn is_chainable(&self) -> bool {
        match self {
            Self::Chainable(_) => true,
            Self::NotChainable(_) => false,
        }
    }
}

impl IntoIterator for CoverageSuccessors<'_> {
    type Item = BasicBlock;
    type IntoIter = impl DoubleEndedIterator<Item = Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        match self {
            Self::Chainable(bb) => Some(bb).into_iter().chain((&[]).iter().copied()),
            Self::NotChainable(bbs) => None.into_iter().chain(bbs.iter().copied()),
        }
    }
}

// Returns the subset of a block's successors that are relevant to the coverage
// graph, i.e. those that do not represent unwinds or false edges.
// FIXME(#78544): MIR InstrumentCoverage: Improve coverage of `#[should_panic]` tests and
// `catch_unwind()` handlers.
fn bcb_filtered_successors<'a, 'tcx>(terminator: &'a Terminator<'tcx>) -> CoverageSuccessors<'a> {
    use TerminatorKind::*;
    match terminator.kind {
        // A switch terminator can have many coverage-relevant successors.
        // (If there is exactly one successor, we still treat it as not chainable.)
        SwitchInt { ref targets, .. } => CoverageSuccessors::NotChainable(targets.all_targets()),

        // A yield terminator has exactly 1 successor, but should not be chained,
        // because its resume edge has a different execution count.
        Yield { ref resume, .. } => CoverageSuccessors::NotChainable(std::slice::from_ref(resume)),

        // These terminators have exactly one coverage-relevant successor,
        // and can be chained into it.
        Assert { target, .. }
        | Drop { target, .. }
        | FalseEdge { real_target: target, .. }
        | FalseUnwind { real_target: target, .. }
        | Goto { target } => CoverageSuccessors::Chainable(target),

        // These terminators can normally be chained, except when they have no
        // successor because they are known to diverge.
        Call { target: maybe_target, .. } | InlineAsm { destination: maybe_target, .. } => {
            match maybe_target {
                Some(target) => CoverageSuccessors::Chainable(target),
                None => CoverageSuccessors::NotChainable(&[]),
            }
        }

        // These terminators have no coverage-relevant successors.
        CoroutineDrop | Return | Unreachable | UnwindResume | UnwindTerminate(_) => {
            CoverageSuccessors::NotChainable(&[])
        }
    }
}

/// Maintains separate worklists for each loop in the BasicCoverageBlock CFG, plus one for the
/// CoverageGraph outside all loops. This supports traversing the BCB CFG in a way that
/// ensures a loop is completely traversed before processing Blocks after the end of the loop.
#[derive(Debug)]
pub(super) struct TraversalContext {
    /// BCB with one or more incoming loop backedges, indicating which loop
    /// this context is for.
    ///
    /// If `None`, this is the non-loop context for the function as a whole.
    loop_header: Option<BasicCoverageBlock>,

    /// Worklist of BCBs to be processed in this context.
    worklist: VecDeque<BasicCoverageBlock>,
}

pub(super) struct TraverseCoverageGraphWithLoops<'a> {
    basic_coverage_blocks: &'a CoverageGraph,

    backedges: IndexVec<BasicCoverageBlock, Vec<BasicCoverageBlock>>,
    context_stack: Vec<TraversalContext>,
    visited: BitSet<BasicCoverageBlock>,
}

impl<'a> TraverseCoverageGraphWithLoops<'a> {
    pub(super) fn new(basic_coverage_blocks: &'a CoverageGraph) -> Self {
        let backedges = find_loop_backedges(basic_coverage_blocks);

        let worklist = VecDeque::from([basic_coverage_blocks.start_node()]);
        let context_stack = vec![TraversalContext { loop_header: None, worklist }];

        // `context_stack` starts with a `TraversalContext` for the main function context (beginning
        // with the `start` BasicCoverageBlock of the function). New worklists are pushed to the top
        // of the stack as loops are entered, and popped off of the stack when a loop's worklist is
        // exhausted.
        let visited = BitSet::new_empty(basic_coverage_blocks.num_nodes());
        Self { basic_coverage_blocks, backedges, context_stack, visited }
    }

    /// For each loop on the loop context stack (top-down), yields a list of BCBs
    /// within that loop that have an outgoing edge back to the loop header.
    pub(super) fn reloop_bcbs_per_loop(&self) -> impl Iterator<Item = &[BasicCoverageBlock]> {
        self.context_stack
            .iter()
            .rev()
            .filter_map(|context| context.loop_header)
            .map(|header_bcb| self.backedges[header_bcb].as_slice())
    }

    pub(super) fn next(&mut self) -> Option<BasicCoverageBlock> {
        debug!(
            "TraverseCoverageGraphWithLoops::next - context_stack: {:?}",
            self.context_stack.iter().rev().collect::<Vec<_>>()
        );

        while let Some(context) = self.context_stack.last_mut() {
            if let Some(bcb) = context.worklist.pop_front() {
                if !self.visited.insert(bcb) {
                    debug!("Already visited: {bcb:?}");
                    continue;
                }
                debug!("Visiting {bcb:?}");

                if self.backedges[bcb].len() > 0 {
                    debug!("{bcb:?} is a loop header! Start a new TraversalContext...");
                    self.context_stack.push(TraversalContext {
                        loop_header: Some(bcb),
                        worklist: VecDeque::new(),
                    });
                }
                self.add_successors_to_worklists(bcb);
                return Some(bcb);
            } else {
                // Strip contexts with empty worklists from the top of the stack
                self.context_stack.pop();
            }
        }

        None
    }

    pub fn add_successors_to_worklists(&mut self, bcb: BasicCoverageBlock) {
        let successors = &self.basic_coverage_blocks.successors[bcb];
        debug!("{:?} has {} successors:", bcb, successors.len());

        for &successor in successors {
            if successor == bcb {
                debug!(
                    "{:?} has itself as its own successor. (Note, the compiled code will \
                    generate an infinite loop.)",
                    bcb
                );
                // Don't re-add this successor to the worklist. We are already processing it.
                // FIXME: This claims to skip just the self-successor, but it actually skips
                // all other successors as well. Does that matter?
                break;
            }

            // Add successors of the current BCB to the appropriate context. Successors that
            // stay within a loop are added to the BCBs context worklist. Successors that
            // exit the loop (they are not dominated by the loop header) must be reachable
            // from other BCBs outside the loop, and they will be added to a different
            // worklist.
            //
            // Branching blocks (with more than one successor) must be processed before
            // blocks with only one successor, to prevent unnecessarily complicating
            // `Expression`s by creating a Counter in a `BasicCoverageBlock` that the
            // branching block would have given an `Expression` (or vice versa).

            let context = self
                .context_stack
                .iter_mut()
                .rev()
                .find(|context| match context.loop_header {
                    Some(loop_header) => {
                        self.basic_coverage_blocks.dominates(loop_header, successor)
                    }
                    None => true,
                })
                .unwrap_or_else(|| bug!("should always fall back to the root non-loop context"));
            debug!("adding to worklist for {:?}", context.loop_header);

            // FIXME: The code below had debug messages claiming to add items to a
            // particular end of the worklist, but was confused about which end was
            // which. The existing behaviour has been preserved for now, but it's
            // unclear what the intended behaviour was.

            if self.basic_coverage_blocks.successors[successor].len() > 1 {
                context.worklist.push_back(successor);
            } else {
                context.worklist.push_front(successor);
            }
        }
    }

    pub fn is_complete(&self) -> bool {
        self.visited.count() == self.visited.domain_size()
    }

    pub fn unvisited(&self) -> Vec<BasicCoverageBlock> {
        let mut unvisited_set: BitSet<BasicCoverageBlock> =
            BitSet::new_filled(self.visited.domain_size());
        unvisited_set.subtract(&self.visited);
        unvisited_set.iter().collect::<Vec<_>>()
    }
}

pub(super) fn find_loop_backedges(
    basic_coverage_blocks: &CoverageGraph,
) -> IndexVec<BasicCoverageBlock, Vec<BasicCoverageBlock>> {
    let num_bcbs = basic_coverage_blocks.num_nodes();
    let mut backedges = IndexVec::from_elem_n(Vec::<BasicCoverageBlock>::new(), num_bcbs);

    // Identify loops by their backedges.
    for (bcb, _) in basic_coverage_blocks.iter_enumerated() {
        for &successor in &basic_coverage_blocks.successors[bcb] {
            if basic_coverage_blocks.dominates(successor, bcb) {
                let loop_header = successor;
                let backedge_from_bcb = bcb;
                debug!(
                    "Found BCB backedge: {:?} -> loop_header: {:?}",
                    backedge_from_bcb, loop_header
                );
                backedges[loop_header].push(backedge_from_bcb);
            }
        }
    }
    backedges
}

fn short_circuit_preorder<'a, 'tcx, F, Iter>(
    body: &'a mir::Body<'tcx>,
    filtered_successors: F,
) -> impl Iterator<Item = BasicBlock> + Captures<'a> + Captures<'tcx>
where
    F: Fn(BasicBlock) -> Iter,
    Iter: IntoIterator<Item = BasicBlock>,
{
    let mut visited = BitSet::new_empty(body.basic_blocks.len());
    let mut worklist = vec![mir::START_BLOCK];

    std::iter::from_fn(move || {
        while let Some(bb) = worklist.pop() {
            if !visited.insert(bb) {
                continue;
            }

            worklist.extend(filtered_successors(bb));

            return Some(bb);
        }

        None
    })
}
