use rustc_data_structures::captures::Captures;
use rustc_data_structures::graph::dominators::{self, Dominators};
use rustc_data_structures::graph::{self, GraphSuccessors, WithNumNodes, WithStartNode};
use rustc_index::bit_set::BitSet;
use rustc_index::{IndexSlice, IndexVec};
use rustc_middle::mir::{self, BasicBlock, TerminatorKind};

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

        let mut seen = IndexVec::from_elem(false, &bcbs);
        let successors = IndexVec::from_fn_n(
            |bcb| {
                for b in seen.iter_mut() {
                    *b = false;
                }
                let bcb_data = &bcbs[bcb];
                let mut bcb_successors = Vec::new();
                for successor in bcb_filtered_successors(&mir_body, bcb_data.last_bb())
                    .filter_map(|successor_bb| bb_to_bcb[successor_bb])
                {
                    if !seen[successor] {
                        seen[successor] = true;
                        bcb_successors.push(successor);
                    }
                }
                bcb_successors
            },
            bcbs.len(),
        );

        let mut predecessors = IndexVec::from_elem(Vec::new(), &bcbs);
        for (bcb, bcb_successors) in successors.iter_enumerated() {
            for &successor in bcb_successors {
                predecessors[successor].push(bcb);
            }
        }

        let mut basic_coverage_blocks =
            Self { bcbs, bb_to_bcb, successors, predecessors, dominators: None };
        let dominators = dominators::dominators(&basic_coverage_blocks);
        basic_coverage_blocks.dominators = Some(dominators);
        basic_coverage_blocks
    }

    fn compute_basic_coverage_blocks(
        mir_body: &mir::Body<'_>,
    ) -> (
        IndexVec<BasicCoverageBlock, BasicCoverageBlockData>,
        IndexVec<BasicBlock, Option<BasicCoverageBlock>>,
    ) {
        let num_basic_blocks = mir_body.basic_blocks.len();
        let mut bcbs = IndexVec::with_capacity(num_basic_blocks);
        let mut bb_to_bcb = IndexVec::from_elem_n(None, num_basic_blocks);

        // Walk the MIR CFG using a Preorder traversal, which starts from `START_BLOCK` and follows
        // each block terminator's `successors()`. Coverage spans must map to actual source code,
        // so compiler generated blocks and paths can be ignored. To that end, the CFG traversal
        // intentionally omits unwind paths.
        // FIXME(#78544): MIR InstrumentCoverage: Improve coverage of `#[should_panic]` tests and
        // `catch_unwind()` handlers.

        let mut basic_blocks = Vec::new();
        for bb in short_circuit_preorder(mir_body, bcb_filtered_successors) {
            if let Some(last) = basic_blocks.last() {
                let predecessors = &mir_body.basic_blocks.predecessors()[bb];
                if predecessors.len() > 1 || !predecessors.contains(last) {
                    // The `bb` has more than one _incoming_ edge, and should start its own
                    // `BasicCoverageBlockData`. (Note, the `basic_blocks` vector does not yet
                    // include `bb`; it contains a sequence of one or more sequential basic_blocks
                    // with no intermediate branches in or out. Save these as a new
                    // `BasicCoverageBlockData` before starting the new one.)
                    Self::add_basic_coverage_block(
                        &mut bcbs,
                        &mut bb_to_bcb,
                        basic_blocks.split_off(0),
                    );
                    debug!(
                        "  because {}",
                        if predecessors.len() > 1 {
                            "predecessors.len() > 1".to_owned()
                        } else {
                            format!("bb {} is not in predecessors: {:?}", bb.index(), predecessors)
                        }
                    );
                }
            }
            basic_blocks.push(bb);

            let term = mir_body[bb].terminator();

            match term.kind {
                TerminatorKind::Return { .. }
                | TerminatorKind::UnwindTerminate(_)
                | TerminatorKind::Yield { .. }
                | TerminatorKind::SwitchInt { .. } => {
                    // The `bb` has more than one _outgoing_ edge, or exits the function. Save the
                    // current sequence of `basic_blocks` gathered to this point, as a new
                    // `BasicCoverageBlockData`.
                    Self::add_basic_coverage_block(
                        &mut bcbs,
                        &mut bb_to_bcb,
                        basic_blocks.split_off(0),
                    );
                    debug!("  because term.kind = {:?}", term.kind);
                    // Note that this condition is based on `TerminatorKind`, even though it
                    // theoretically boils down to `successors().len() != 1`; that is, either zero
                    // (e.g., `Return`, `Terminate`) or multiple successors (e.g., `SwitchInt`), but
                    // since the BCB CFG ignores things like unwind branches (which exist in the
                    // `Terminator`s `successors()` list) checking the number of successors won't
                    // work.
                }

                // The following `TerminatorKind`s are either not expected outside an unwind branch,
                // or they should not (under normal circumstances) branch. Coverage graphs are
                // simplified by assuring coverage results are accurate for program executions that
                // don't panic.
                //
                // Programs that panic and unwind may record slightly inaccurate coverage results
                // for a coverage region containing the `Terminator` that began the panic. This
                // is as intended. (See Issue #78544 for a possible future option to support
                // coverage in test programs that panic.)
                TerminatorKind::Goto { .. }
                | TerminatorKind::UnwindResume
                | TerminatorKind::Unreachable
                | TerminatorKind::Drop { .. }
                | TerminatorKind::Call { .. }
                | TerminatorKind::CoroutineDrop
                | TerminatorKind::Assert { .. }
                | TerminatorKind::FalseEdge { .. }
                | TerminatorKind::FalseUnwind { .. }
                | TerminatorKind::InlineAsm { .. } => {}
            }
        }

        if !basic_blocks.is_empty() {
            // process any remaining basic_blocks into a final `BasicCoverageBlockData`
            Self::add_basic_coverage_block(&mut bcbs, &mut bb_to_bcb, basic_blocks.split_off(0));
            debug!("  because the end of the MIR CFG was reached while traversing");
        }

        (bcbs, bb_to_bcb)
    }

    fn add_basic_coverage_block(
        bcbs: &mut IndexVec<BasicCoverageBlock, BasicCoverageBlockData>,
        bb_to_bcb: &mut IndexSlice<BasicBlock, Option<BasicCoverageBlock>>,
        basic_blocks: Vec<BasicBlock>,
    ) {
        let bcb = bcbs.next_index();
        for &bb in basic_blocks.iter() {
            bb_to_bcb[bb] = Some(bcb);
        }
        let bcb_data = BasicCoverageBlockData::from(basic_blocks);
        debug!("adding bcb{}: {:?}", bcb.index(), bcb_data);
        bcbs.push(bcb_data);
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

/// Represents a successor from a branching BasicCoverageBlock (such as the arms of a `SwitchInt`)
/// as either the successor BCB itself, if it has only one incoming edge, or the successor _plus_
/// the specific branching BCB, representing the edge between the two. The latter case
/// distinguishes this incoming edge from other incoming edges to the same `target_bcb`.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) struct BcbBranch {
    pub edge_from_bcb: Option<BasicCoverageBlock>,
    pub target_bcb: BasicCoverageBlock,
}

impl BcbBranch {
    pub fn from_to(
        from_bcb: BasicCoverageBlock,
        to_bcb: BasicCoverageBlock,
        basic_coverage_blocks: &CoverageGraph,
    ) -> Self {
        let edge_from_bcb = if basic_coverage_blocks.predecessors[to_bcb].len() > 1 {
            Some(from_bcb)
        } else {
            None
        };
        Self { edge_from_bcb, target_bcb: to_bcb }
    }

    pub fn is_only_path_to_target(&self) -> bool {
        self.edge_from_bcb.is_none()
    }
}

impl std::fmt::Debug for BcbBranch {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(from_bcb) = self.edge_from_bcb {
            write!(fmt, "{:?}->{:?}", from_bcb, self.target_bcb)
        } else {
            write!(fmt, "{:?}", self.target_bcb)
        }
    }
}

// Returns the subset of a block's successors that are relevant to the coverage
// graph, i.e. those that do not represent unwinds or unreachable branches.
// FIXME(#78544): MIR InstrumentCoverage: Improve coverage of `#[should_panic]` tests and
// `catch_unwind()` handlers.
fn bcb_filtered_successors<'a, 'tcx>(
    body: &'a mir::Body<'tcx>,
    bb: BasicBlock,
) -> impl Iterator<Item = BasicBlock> + Captures<'a> + Captures<'tcx> {
    let terminator = body[bb].terminator();

    let take_n_successors = match terminator.kind {
        // SwitchInt successors are never unwinds, so all of them should be traversed.
        TerminatorKind::SwitchInt { .. } => usize::MAX,
        // For all other kinds, return only the first successor (if any), ignoring any
        // unwind successors.
        _ => 1,
    };

    terminator
        .successors()
        .take(take_n_successors)
        .filter(move |&successor| body[successor].terminator().kind != TerminatorKind::Unreachable)
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
    F: Fn(&'a mir::Body<'tcx>, BasicBlock) -> Iter,
    Iter: Iterator<Item = BasicBlock>,
{
    let mut visited = BitSet::new_empty(body.basic_blocks.len());
    let mut worklist = vec![mir::START_BLOCK];

    std::iter::from_fn(move || {
        while let Some(bb) = worklist.pop() {
            if !visited.insert(bb) {
                continue;
            }

            worklist.extend(filtered_successors(body, bb));

            return Some(bb);
        }

        None
    })
}
