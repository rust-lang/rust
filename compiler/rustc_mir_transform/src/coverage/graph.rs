use std::cmp::Ordering;
use std::collections::VecDeque;
use std::iter;
use std::ops::{Index, IndexMut};

use rustc_data_structures::captures::Captures;
use rustc_data_structures::fx::FxHashSet;
use rustc_data_structures::graph::dominators::Dominators;
use rustc_data_structures::graph::{self, DirectedGraph, StartNode};
use rustc_index::IndexVec;
use rustc_index::bit_set::BitSet;
use rustc_middle::bug;
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
    is_loop_header: BitSet<BasicCoverageBlock>,
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

        let num_nodes = bcbs.len();
        let mut this = Self {
            bcbs,
            bb_to_bcb,
            successors,
            predecessors,
            dominators: None,
            dominator_order_rank: IndexVec::from_elem_n(0, num_nodes),
            is_loop_header: BitSet::new_empty(num_nodes),
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

        let mut add_basic_coverage_block = |basic_blocks: &mut Vec<BasicBlock>| {
            // Take the accumulated list of blocks, leaving the vector empty
            // to be used by subsequent BCBs.
            let basic_blocks = std::mem::take(basic_blocks);

            let bcb = bcbs.next_index();
            for &bb in basic_blocks.iter() {
                bb_to_bcb[bb] = Some(bcb);
            }

            let is_out_summable = basic_blocks.last().map_or(false, |&bb| {
                bcb_filtered_successors(mir_body[bb].terminator()).is_out_summable()
            });
            let bcb_data = BasicCoverageBlockData { basic_blocks, is_out_summable };
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

    /// Returns the source of this node's sole in-edge, if it has exactly one.
    /// That edge can be assumed to have the same execution count as the node
    /// itself (in the absence of panics).
    pub(crate) fn sole_predecessor(
        &self,
        to_bcb: BasicCoverageBlock,
    ) -> Option<BasicCoverageBlock> {
        // Unlike `simple_successor`, there is no need for extra checks here.
        if let &[from_bcb] = self.predecessors[to_bcb].as_slice() { Some(from_bcb) } else { None }
    }

    /// Returns the target of this node's sole out-edge, if it has exactly
    /// one, but only if that edge can be assumed to have the same execution
    /// count as the node itself (in the absence of panics).
    pub(crate) fn simple_successor(
        &self,
        from_bcb: BasicCoverageBlock,
    ) -> Option<BasicCoverageBlock> {
        // If a node's count is the sum of its out-edges, and it has exactly
        // one out-edge, then that edge has the same count as the node.
        if self.bcbs[from_bcb].is_out_summable
            && let &[to_bcb] = self.successors[from_bcb].as_slice()
        {
            Some(to_bcb)
        } else {
            None
        }
    }

    /// For each loop that contains the given node, yields the "loop header"
    /// node representing that loop, from innermost to outermost. If the given
    /// node is itself a loop header, it is yielded first.
    pub(crate) fn loop_headers_containing(
        &self,
        bcb: BasicCoverageBlock,
    ) -> impl Iterator<Item = BasicCoverageBlock> + Captures<'_> {
        let self_if_loop_header = self.is_loop_header.contains(bcb).then_some(bcb).into_iter();

        let mut curr = Some(bcb);
        let strictly_enclosing = iter::from_fn(move || {
            let enclosing = self.enclosing_loop_header[curr?];
            curr = enclosing;
            enclosing
        });

        self_if_loop_header.chain(strictly_enclosing)
    }

    /// For the given node, yields the subset of its predecessor nodes that
    /// it dominates. If that subset is non-empty, the node is a "loop header",
    /// and each of those predecessors represents an in-edge that jumps back to
    /// the top of its loop.
    pub(crate) fn reloop_predecessors(
        &self,
        to_bcb: BasicCoverageBlock,
    ) -> impl Iterator<Item = BasicCoverageBlock> + Captures<'_> {
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

rustc_index::newtype_index! {
    /// A node in the control-flow graph of CoverageGraph.
    #[orderable]
    #[debug_format = "bcb{}"]
    pub(crate) struct BasicCoverageBlock {
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
enum CoverageSuccessors<'a> {
    /// The terminator has exactly one straight-line successor, so its block can
    /// potentially be combined into the same BCB as that successor.
    Chainable(BasicBlock),
    /// The block cannot be combined into the same BCB as its successor(s).
    NotChainable(&'a [BasicBlock]),
    /// Yield terminators are not chainable, and their execution count can also
    /// differ from the execution count of their out-edge.
    Yield(BasicBlock),
}

impl CoverageSuccessors<'_> {
    fn is_chainable(&self) -> bool {
        match self {
            Self::Chainable(_) => true,
            Self::NotChainable(_) => false,
            Self::Yield(_) => false,
        }
    }

    /// Returns true if the terminator itself is assumed to have the same
    /// execution count as the sum of its out-edges (assuming no panics).
    fn is_out_summable(&self) -> bool {
        match self {
            Self::Chainable(_) => true,
            Self::NotChainable(_) => true,
            Self::Yield(_) => false,
        }
    }
}

impl IntoIterator for CoverageSuccessors<'_> {
    type Item = BasicBlock;
    type IntoIter = impl DoubleEndedIterator<Item = Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        match self {
            Self::Chainable(bb) | Self::Yield(bb) => {
                Some(bb).into_iter().chain((&[]).iter().copied())
            }
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
        Yield { resume, .. } => CoverageSuccessors::Yield(resume),

        // These terminators have exactly one coverage-relevant successor,
        // and can be chained into it.
        Assert { target, .. }
        | Drop { target, .. }
        | FalseEdge { real_target: target, .. }
        | FalseUnwind { real_target: target, .. }
        | Goto { target } => CoverageSuccessors::Chainable(target),

        // A call terminator can normally be chained, except when it has no
        // successor because it is known to diverge.
        Call { target: maybe_target, .. } => match maybe_target {
            Some(target) => CoverageSuccessors::Chainable(target),
            None => CoverageSuccessors::NotChainable(&[]),
        },

        // An inline asm terminator can normally be chained, except when it
        // diverges or uses asm goto.
        InlineAsm { ref targets, .. } => {
            if let [target] = targets[..] {
                CoverageSuccessors::Chainable(target)
            } else {
                CoverageSuccessors::NotChainable(targets)
            }
        }

        // These terminators have no coverage-relevant successors.
        CoroutineDrop
        | Return
        | TailCall { .. }
        | Unreachable
        | UnwindResume
        | UnwindTerminate(_) => CoverageSuccessors::NotChainable(&[]),
    }
}

/// Maintains separate worklists for each loop in the BasicCoverageBlock CFG, plus one for the
/// CoverageGraph outside all loops. This supports traversing the BCB CFG in a way that
/// ensures a loop is completely traversed before processing Blocks after the end of the loop.
#[derive(Debug)]
struct TraversalContext {
    /// BCB with one or more incoming loop backedges, indicating which loop
    /// this context is for.
    ///
    /// If `None`, this is the non-loop context for the function as a whole.
    loop_header: Option<BasicCoverageBlock>,

    /// Worklist of BCBs to be processed in this context.
    worklist: VecDeque<BasicCoverageBlock>,
}

pub(crate) struct TraverseCoverageGraphWithLoops<'a> {
    basic_coverage_blocks: &'a CoverageGraph,

    context_stack: Vec<TraversalContext>,
    visited: BitSet<BasicCoverageBlock>,
}

impl<'a> TraverseCoverageGraphWithLoops<'a> {
    pub(crate) fn new(basic_coverage_blocks: &'a CoverageGraph) -> Self {
        let worklist = VecDeque::from([basic_coverage_blocks.start_node()]);
        let context_stack = vec![TraversalContext { loop_header: None, worklist }];

        // `context_stack` starts with a `TraversalContext` for the main function context (beginning
        // with the `start` BasicCoverageBlock of the function). New worklists are pushed to the top
        // of the stack as loops are entered, and popped off of the stack when a loop's worklist is
        // exhausted.
        let visited = BitSet::new_empty(basic_coverage_blocks.num_nodes());
        Self { basic_coverage_blocks, context_stack, visited }
    }

    pub(crate) fn next(&mut self) -> Option<BasicCoverageBlock> {
        debug!(
            "TraverseCoverageGraphWithLoops::next - context_stack: {:?}",
            self.context_stack.iter().rev().collect::<Vec<_>>()
        );

        while let Some(context) = self.context_stack.last_mut() {
            let Some(bcb) = context.worklist.pop_front() else {
                // This stack level is exhausted; pop it and try the next one.
                self.context_stack.pop();
                continue;
            };

            if !self.visited.insert(bcb) {
                debug!("Already visited: {bcb:?}");
                continue;
            }
            debug!("Visiting {bcb:?}");

            if self.basic_coverage_blocks.is_loop_header.contains(bcb) {
                debug!("{bcb:?} is a loop header! Start a new TraversalContext...");
                self.context_stack
                    .push(TraversalContext { loop_header: Some(bcb), worklist: VecDeque::new() });
            }
            self.add_successors_to_worklists(bcb);
            return Some(bcb);
        }

        None
    }

    fn add_successors_to_worklists(&mut self, bcb: BasicCoverageBlock) {
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

    pub(crate) fn is_complete(&self) -> bool {
        self.visited.count() == self.visited.domain_size()
    }

    pub(crate) fn unvisited(&self) -> Vec<BasicCoverageBlock> {
        let mut unvisited_set: BitSet<BasicCoverageBlock> =
            BitSet::new_filled(self.visited.domain_size());
        unvisited_set.subtract(&self.visited);
        unvisited_set.iter().collect::<Vec<_>>()
    }
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
