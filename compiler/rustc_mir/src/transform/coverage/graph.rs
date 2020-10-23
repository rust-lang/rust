use rustc_data_structures::graph::dominators::{self, Dominators};
use rustc_data_structures::graph::{self, GraphSuccessors, WithNumNodes};
use rustc_index::bit_set::BitSet;
use rustc_index::vec::IndexVec;
use rustc_middle::mir::{self, BasicBlock, BasicBlockData, Terminator, TerminatorKind};

use std::ops::{Index, IndexMut};

const ID_SEPARATOR: &str = ",";

/// A coverage-specific simplification of the MIR control flow graph (CFG). The `CoverageGraph`s
/// nodes are `BasicCoverageBlock`s, which encompass one or more MIR `BasicBlock`s, plus a
/// `CoverageKind` counter (to be added by `CoverageCounters::make_bcb_counters`), and an optional
/// set of additional counters--if needed--to count incoming edges, if there are more than one.
/// (These "edge counters" are eventually converted into new MIR `BasicBlock`s.)
pub(crate) struct CoverageGraph {
    bcbs: IndexVec<BasicCoverageBlock, BasicCoverageBlockData>,
    bb_to_bcb: IndexVec<BasicBlock, Option<BasicCoverageBlock>>,
    pub successors: IndexVec<BasicCoverageBlock, Vec<BasicCoverageBlock>>,
    pub predecessors: IndexVec<BasicCoverageBlock, Vec<BasicCoverageBlock>>,
    dominators: Option<Dominators<BasicCoverageBlock>>,
}

impl CoverageGraph {
    pub fn from_mir(mir_body: &mir::Body<'tcx>) -> Self {
        let (bcbs, bb_to_bcb) = Self::compute_basic_coverage_blocks(mir_body);

        // Pre-transform MIR `BasicBlock` successors and predecessors into the BasicCoverageBlock
        // equivalents. Note that since the BasicCoverageBlock graph has been fully simplified, the
        // each predecessor of a BCB leader_bb should be in a unique BCB, and each successor of a
        // BCB last_bb should bin in its own unique BCB. Therefore, collecting the BCBs using
        // `bb_to_bcb` should work without requiring a deduplication step.

        let successors = IndexVec::from_fn_n(
            |bcb| {
                let bcb_data = &bcbs[bcb];
                let bcb_successors =
                    bcb_filtered_successors(&mir_body, &bcb_data.terminator(mir_body).kind)
                        .filter_map(|&successor_bb| bb_to_bcb[successor_bb])
                        .collect::<Vec<_>>();
                debug_assert!({
                    let mut sorted = bcb_successors.clone();
                    sorted.sort_unstable();
                    let initial_len = sorted.len();
                    sorted.dedup();
                    sorted.len() == initial_len
                });
                bcb_successors
            },
            bcbs.len(),
        );

        let mut predecessors = IndexVec::from_elem_n(Vec::new(), bcbs.len());
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
        mir_body: &mir::Body<'tcx>,
    ) -> (
        IndexVec<BasicCoverageBlock, BasicCoverageBlockData>,
        IndexVec<BasicBlock, Option<BasicCoverageBlock>>,
    ) {
        let num_basic_blocks = mir_body.num_nodes();
        let mut bcbs = IndexVec::with_capacity(num_basic_blocks);
        let mut bb_to_bcb = IndexVec::from_elem_n(None, num_basic_blocks);

        // Walk the MIR CFG using a Preorder traversal, which starts from `START_BLOCK` and follows
        // each block terminator's `successors()`. Coverage spans must map to actual source code,
        // so compiler generated blocks and paths can be ignored. To that end, the CFG traversal
        // intentionally omits unwind paths.
        let mir_cfg_without_unwind = ShortCircuitPreorder::new(&mir_body, bcb_filtered_successors);

        let mut basic_blocks = Vec::new();
        for (bb, data) in mir_cfg_without_unwind {
            if let Some(last) = basic_blocks.last() {
                let predecessors = &mir_body.predecessors()[bb];
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
                            format!("bb {} is not in precessors: {:?}", bb.index(), predecessors)
                        }
                    );
                }
            }
            basic_blocks.push(bb);

            let term = data.terminator();

            match term.kind {
                TerminatorKind::Return { .. }
                // FIXME(richkadel): Add test(s) for `Abort` coverage.
                | TerminatorKind::Abort
                // FIXME(richkadel): Add test(s) for `Assert` coverage.
                // Should `Assert` be handled like `FalseUnwind` instead? Since we filter out unwind
                // branches when creating the BCB CFG, aren't `Assert`s (without unwinds) just like
                // `FalseUnwinds` (which are kind of like `Goto`s)?
                | TerminatorKind::Assert { .. }
                // FIXME(richkadel): Add test(s) for `Yield` coverage, and confirm coverage is
                // sensible for code using the `yield` keyword.
                | TerminatorKind::Yield { .. }
                // FIXME(richkadel): Also add coverage tests using async/await, and threading.

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
                    // (e.g., `Return`, `Abort`) or multiple successors (e.g., `SwitchInt`), but
                    // since the BCB CFG ignores things like unwind branches (which exist in the
                    // `Terminator`s `successors()` list) checking the number of successors won't
                    // work.
                }
                TerminatorKind::Goto { .. }
                | TerminatorKind::Resume
                | TerminatorKind::Unreachable
                | TerminatorKind::Drop { .. }
                | TerminatorKind::DropAndReplace { .. }
                | TerminatorKind::Call { .. }
                | TerminatorKind::GeneratorDrop
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
        bb_to_bcb: &mut IndexVec<BasicBlock, Option<BasicCoverageBlock>>,
        basic_blocks: Vec<BasicBlock>,
    ) {
        let bcb = BasicCoverageBlock::from_usize(bcbs.len());
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
    pub fn is_dominated_by(&self, node: BasicCoverageBlock, dom: BasicCoverageBlock) -> bool {
        self.dominators.as_ref().unwrap().is_dominated_by(node, dom)
    }

    #[inline(always)]
    pub fn dominators(&self) -> &Dominators<BasicCoverageBlock> {
        self.dominators.as_ref().unwrap()
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

impl graph::GraphPredecessors<'graph> for CoverageGraph {
    type Item = BasicCoverageBlock;
    type Iter = std::vec::IntoIter<BasicCoverageBlock>;
}

impl graph::WithPredecessors for CoverageGraph {
    #[inline]
    fn predecessors(&self, node: Self::Node) -> <Self as graph::GraphPredecessors<'_>>::Iter {
        self.predecessors[node].clone().into_iter()
    }
}

rustc_index::newtype_index! {
    /// A node in the [control-flow graph][CFG] of CoverageGraph.
    pub(crate) struct BasicCoverageBlock {
        DEBUG_FORMAT = "bcb{}",
    }
}

/// A BasicCoverageBlockData (BCB) represents the maximal-length sequence of MIR BasicBlocks without
/// conditional branches, and form a new, simplified, coverage-specific Control Flow Graph, without
/// altering the original MIR CFG.
///
/// Note that running the MIR `SimplifyCfg` transform is not sufficient (and therefore not
/// necessary). The BCB-based CFG is a more aggressive simplification. For example:
///
///   * The BCB CFG ignores (trims) branches not relevant to coverage, such as unwind-related code,
///     that is injected by the Rust compiler but has no physical source code to count. This also
///     means a BasicBlock with a `Call` terminator can be merged into its primary successor target
///     block, in the same BCB.
///   * Some BasicBlock terminators support Rust-specific concerns--like borrow-checking--that are
///     not relevant to coverage analysis. `FalseUnwind`, for example, can be treated the same as
///     a `Goto`, and merged with its successor into the same BCB.
///
/// Each BCB with at least one computed `CoverageSpan` will have no more than one `Counter`.
/// In some cases, a BCB's execution count can be computed by `Expression`. Additional
/// disjoint `CoverageSpan`s in a BCB can also be counted by `Expression` (by adding `ZERO`
/// to the BCB's primary counter or expression).
///
/// The BCB CFG is critical to simplifying the coverage analysis by ensuring graph path-based
/// queries (`is_dominated_by()`, `predecessors`, `successors`, etc.) have branch (control flow)
/// significance.
#[derive(Debug, Clone)]
pub(crate) struct BasicCoverageBlockData {
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

    #[inline(always)]
    pub fn terminator<'a, 'tcx>(&self, mir_body: &'a mir::Body<'tcx>) -> &'a Terminator<'tcx> {
        &mir_body[self.last_bb()].terminator()
    }

    pub fn id(&self) -> String {
        format!(
            "@{}",
            self.basic_blocks
                .iter()
                .map(|bb| bb.index().to_string())
                .collect::<Vec<_>>()
                .join(ID_SEPARATOR)
        )
    }
}

fn bcb_filtered_successors<'a, 'tcx>(
    body: &'tcx &'a mir::Body<'tcx>,
    term_kind: &'tcx TerminatorKind<'tcx>,
) -> Box<dyn Iterator<Item = &'a BasicBlock> + 'a> {
    let mut successors = term_kind.successors();
    box match &term_kind {
        // SwitchInt successors are never unwind, and all of them should be traversed.
        TerminatorKind::SwitchInt { .. } => successors,
        // For all other kinds, return only the first successor, if any, and ignore unwinds.
        // NOTE: `chain(&[])` is required to coerce the `option::iter` (from
        // `next().into_iter()`) into the `mir::Successors` aliased type.
        _ => successors.next().into_iter().chain(&[]),
    }
    .filter(move |&&successor| body[successor].terminator().kind != TerminatorKind::Unreachable)
}

pub struct ShortCircuitPreorder<
    'a,
    'tcx,
    F: Fn(
        &'tcx &'a mir::Body<'tcx>,
        &'tcx TerminatorKind<'tcx>,
    ) -> Box<dyn Iterator<Item = &'a BasicBlock> + 'a>,
> {
    body: &'tcx &'a mir::Body<'tcx>,
    visited: BitSet<BasicBlock>,
    worklist: Vec<BasicBlock>,
    filtered_successors: F,
}

impl<
    'a,
    'tcx,
    F: Fn(
        &'tcx &'a mir::Body<'tcx>,
        &'tcx TerminatorKind<'tcx>,
    ) -> Box<dyn Iterator<Item = &'a BasicBlock> + 'a>,
> ShortCircuitPreorder<'a, 'tcx, F>
{
    pub fn new(
        body: &'tcx &'a mir::Body<'tcx>,
        filtered_successors: F,
    ) -> ShortCircuitPreorder<'a, 'tcx, F> {
        let worklist = vec![mir::START_BLOCK];

        ShortCircuitPreorder {
            body,
            visited: BitSet::new_empty(body.basic_blocks().len()),
            worklist,
            filtered_successors,
        }
    }
}

impl<
    'a: 'tcx,
    'tcx,
    F: Fn(
        &'tcx &'a mir::Body<'tcx>,
        &'tcx TerminatorKind<'tcx>,
    ) -> Box<dyn Iterator<Item = &'a BasicBlock> + 'a>,
> Iterator for ShortCircuitPreorder<'a, 'tcx, F>
{
    type Item = (BasicBlock, &'a BasicBlockData<'tcx>);

    fn next(&mut self) -> Option<(BasicBlock, &'a BasicBlockData<'tcx>)> {
        while let Some(idx) = self.worklist.pop() {
            if !self.visited.insert(idx) {
                continue;
            }

            let data = &self.body[idx];

            if let Some(ref term) = data.terminator {
                self.worklist.extend((self.filtered_successors)(&self.body, &term.kind));
            }

            return Some((idx, data));
        }

        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let size = self.body.basic_blocks().len() - self.visited.count();
        (size, Some(size))
    }
}
