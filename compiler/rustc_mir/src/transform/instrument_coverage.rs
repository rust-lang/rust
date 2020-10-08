use crate::transform::MirPass;
use crate::util::pretty;
use crate::util::spanview::{self, SpanViewable};

use rustc_data_structures::fingerprint::Fingerprint;
use rustc_data_structures::graph::dominators::{self, Dominators};
use rustc_data_structures::graph::{self, GraphSuccessors, WithNumNodes, WithStartNode};
use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_data_structures::sync::Lrc;
use rustc_index::bit_set::BitSet;
use rustc_index::vec::IndexVec;
use rustc_middle::hir;
use rustc_middle::hir::map::blocks::FnLikeNode;
use rustc_middle::ich::StableHashingContext;
use rustc_middle::mir;
use rustc_middle::mir::coverage::*;
use rustc_middle::mir::visit::Visitor;
use rustc_middle::mir::{
    AggregateKind, BasicBlock, BasicBlockData, Coverage, CoverageInfo, FakeReadCause, Location,
    Rvalue, SourceInfo, Statement, StatementKind, Terminator, TerminatorKind,
};
use rustc_middle::ty::query::Providers;
use rustc_middle::ty::TyCtxt;
use rustc_span::def_id::DefId;
use rustc_span::source_map::original_sp;
use rustc_span::{BytePos, CharPos, Pos, SourceFile, Span, Symbol, SyntaxContext};

use smallvec::SmallVec;

use std::cmp::Ordering;
use std::ops::{Index, IndexMut};

const ID_SEPARATOR: &str = ",";

/// Inserts `StatementKind::Coverage` statements that either instrument the binary with injected
/// counters, via intrinsic `llvm.instrprof.increment`, and/or inject metadata used during codegen
/// to construct the coverage map.
pub struct InstrumentCoverage;

/// The `query` provider for `CoverageInfo`, requested by `codegen_coverage()` (to inject each
/// counter) and `FunctionCoverage::new()` (to extract the coverage map metadata from the MIR).
pub(crate) fn provide(providers: &mut Providers) {
    providers.coverageinfo = |tcx, def_id| coverageinfo_from_mir(tcx, def_id);
}

/// The `num_counters` argument to `llvm.instrprof.increment` is the max counter_id + 1, or in
/// other words, the number of counter value references injected into the MIR (plus 1 for the
/// reserved `ZERO` counter, which uses counter ID `0` when included in an expression). Injected
/// counters have a counter ID from `1..num_counters-1`.
///
/// `num_expressions` is the number of counter expressions added to the MIR body.
///
/// Both `num_counters` and `num_expressions` are used to initialize new vectors, during backend
/// code generate, to lookup counters and expressions by simple u32 indexes.
///
/// MIR optimization may split and duplicate some BasicBlock sequences, or optimize out some code
/// including injected counters. (It is OK if some counters are optimized out, but those counters
/// are still included in the total `num_counters` or `num_expressions`.) Simply counting the
/// calls may not work; but computing the number of counters or expressions by adding `1` to the
/// highest ID (for a given instrumented function) is valid.
struct CoverageVisitor {
    info: CoverageInfo,
}

impl Visitor<'_> for CoverageVisitor {
    fn visit_coverage(&mut self, coverage: &Coverage, _location: Location) {
        match coverage.kind {
            CoverageKind::Counter { id, .. } => {
                let counter_id = u32::from(id);
                self.info.num_counters = std::cmp::max(self.info.num_counters, counter_id + 1);
            }
            CoverageKind::Expression { id, .. } => {
                let expression_index = u32::MAX - u32::from(id);
                self.info.num_expressions =
                    std::cmp::max(self.info.num_expressions, expression_index + 1);
            }
            _ => {}
        }
    }
}

fn coverageinfo_from_mir<'tcx>(tcx: TyCtxt<'tcx>, def_id: DefId) -> CoverageInfo {
    let mir_body = tcx.optimized_mir(def_id);

    let mut coverage_visitor =
        CoverageVisitor { info: CoverageInfo { num_counters: 0, num_expressions: 0 } };

    coverage_visitor.visit_body(mir_body);
    coverage_visitor.info
}

impl<'tcx> MirPass<'tcx> for InstrumentCoverage {
    fn run_pass(&self, tcx: TyCtxt<'tcx>, mir_body: &mut mir::Body<'tcx>) {
        let mir_source = mir_body.source;

        // If the InstrumentCoverage pass is called on promoted MIRs, skip them.
        // See: https://github.com/rust-lang/rust/pull/73011#discussion_r438317601
        if mir_source.promoted.is_some() {
            trace!(
                "InstrumentCoverage skipped for {:?} (already promoted for Miri evaluation)",
                mir_source.def_id()
            );
            return;
        }

        let hir_id = tcx.hir().local_def_id_to_hir_id(mir_source.def_id().expect_local());
        let is_fn_like = FnLikeNode::from_node(tcx.hir().get(hir_id)).is_some();

        // Only instrument functions, methods, and closures (not constants since they are evaluated
        // at compile time by Miri).
        // FIXME(#73156): Handle source code coverage in const eval, but note, if and when const
        // expressions get coverage spans, we will probably have to "carve out" space for const
        // expressions from coverage spans in enclosing MIR's, like we do for closures. (That might
        // be tricky if const expressions have no corresponding statements in the enclosing MIR.
        // Closures are carved out by their initial `Assign` statement.)
        if !is_fn_like {
            trace!("InstrumentCoverage skipped for {:?} (not an FnLikeNode)", mir_source.def_id());
            return;
        }
        // FIXME(richkadel): By comparison, the MIR pass `ConstProp` includes associated constants,
        // with functions, methods, and closures. I assume Miri is used for associated constants as
        // well. If not, we may need to include them here too.

        trace!("InstrumentCoverage starting for {:?}", mir_source.def_id());
        Instrumentor::new(&self.name(), tcx, mir_body).inject_counters();
        trace!("InstrumentCoverage starting for {:?}", mir_source.def_id());
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
/// In some cases, a BCB's execution count can be computed by `CounterExpression`. Additional
/// disjoint `CoverageSpan`s in a BCB can also be counted by `CounterExpression` (by adding `ZERO`
/// to the BCB's primary counter or expression).
///
/// The BCB CFG is critical to simplifying the coverage analysis by ensuring graph path-based
/// queries (`is_dominated_by()`, `predecessors`, `successors`, etc.) have branch (control flow)
/// significance.
#[derive(Debug, Clone)]
struct BasicCoverageBlockData {
    basic_blocks: Vec<BasicBlock>,
    counter_kind: Option<CoverageKind>,
}

impl BasicCoverageBlockData {
    pub fn from(basic_blocks: Vec<BasicBlock>) -> Self {
        assert!(basic_blocks.len() > 0);
        Self { basic_blocks, counter_kind: None }
    }

    #[inline(always)]
    pub fn basic_blocks(&self) -> std::slice::Iter<'_, BasicBlock> {
        self.basic_blocks.iter()
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
    pub fn terminator<'a, 'tcx>(&self, mir_body: &'a mir::Body<'tcx>) -> &'a TerminatorKind<'tcx> {
        &mir_body[self.last_bb()].terminator().kind
    }

    #[inline(always)]
    pub fn set_counter(&mut self, counter_kind: CoverageKind) {
        self.counter_kind
            .replace(counter_kind)
            .expect_none("attempt to set a BasicCoverageBlock coverage counter more than once");
    }

    #[inline(always)]
    pub fn counter(&self) -> Option<&CoverageKind> {
        self.counter_kind.as_ref()
    }

    #[inline(always)]
    pub fn take_counter(&mut self) -> Option<CoverageKind> {
        self.counter_kind.take()
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

rustc_index::newtype_index! {
    /// A node in the [control-flow graph][CFG] of BasicCoverageBlocks.
    pub struct BasicCoverageBlock {
        DEBUG_FORMAT = "bcb{}",
    }
}

struct BasicCoverageBlocks {
    bcbs: IndexVec<BasicCoverageBlock, BasicCoverageBlockData>,
    bb_to_bcb: IndexVec<BasicBlock, Option<BasicCoverageBlock>>,
    successors: IndexVec<BasicCoverageBlock, Vec<BasicCoverageBlock>>,
    predecessors: IndexVec<BasicCoverageBlock, BcbPredecessors>,
}

impl BasicCoverageBlocks {
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
                let bcb_successors = bcb_data
                    .terminator(mir_body)
                    .successors()
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

        let predecessors = IndexVec::from_fn_n(
            |bcb| {
                let bcb_data = &bcbs[bcb];
                let bcb_predecessors = mir_body.predecessors()[bcb_data.leader_bb()]
                    .iter()
                    .filter_map(|&predecessor_bb| bb_to_bcb[predecessor_bb])
                    .collect::<BcbPredecessors>();
                debug_assert!({
                    let mut sorted = bcb_predecessors.clone();
                    sorted.sort_unstable();
                    let initial_len = sorted.len();
                    sorted.dedup();
                    sorted.len() == initial_len
                });
                bcb_predecessors
            },
            bcbs.len(),
        );

        Self { bcbs, bb_to_bcb, successors, predecessors }
    }

    fn compute_basic_coverage_blocks(
        mir_body: &mir::Body<'tcx>,
    ) -> (
        IndexVec<BasicCoverageBlock, BasicCoverageBlockData>,
        IndexVec<BasicBlock, Option<BasicCoverageBlock>>,
    ) {
        let len = mir_body.num_nodes();
        let mut bcbs = IndexVec::with_capacity(len);
        let mut bb_to_bcb = IndexVec::from_elem_n(None, len);

        // Walk the MIR CFG using a Preorder traversal, which starts from `START_BLOCK` and follows
        // each block terminator's `successors()`. Coverage spans must map to actual source code,
        // so compiler generated blocks and paths can be ignored. To that end, the CFG traversal
        // intentionally omits unwind paths.
        let mir_cfg_without_unwind = ShortCircuitPreorder::new(mir_body, |term_kind| {
            let mut successors = term_kind.successors();
            match &term_kind {
                // SwitchInt successors are never unwind, and all of them should be traversed.
                TerminatorKind::SwitchInt { .. } => successors,
                // For all other kinds, return only the first successor, if any, and ignore unwinds.
                // NOTE: `chain(&[])` is required to coerce the `option::iter` (from
                // `next().into_iter()`) into the `mir::Successors` aliased type.
                _ => successors.next().into_iter().chain(&[]),
            }
        });

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
                    // TODO(richkadel): uncomment debug!
                    // debug!(
                    //     "  because {}",
                    //     if predecessors.len() > 1 {
                    //         "predecessors.len() > 1".to_owned()
                    //     } else {
                    //         format!("bb {} is not in precessors: {:?}", bb.index(), predecessors)
                    //     }
                    // );
                }
            }
            basic_blocks.push(bb);

            let term = data.terminator();

            match term.kind {
                TerminatorKind::Return { .. }
// TODO(richkadel): Do we handle Abort like Return? The program doesn't continue
// normally. It's like a failed assert (I assume).
                | TerminatorKind::Abort
// TODO(richkadel): I think Assert should be handled like falseUnwind.
// It's just a goto if we assume it does not fail the assert, and if it
// does fail, the unwind caused by failure is hidden code not covered
// in coverage???
//
// BUT if we take it out, then we can't count code up until the assert.
//
// (That may be OK, not sure. Maybe not.).
//
// How am I handling the try "?" on functions that return Result?

// TODO(richkadel): comment out?
                | TerminatorKind::Assert { .. }

// TODO(richkadel): And I don't know what do do with Yield
                | TerminatorKind::Yield { .. }
                // FIXME(richkadel): Add coverage test for TerminatorKind::Yield and/or `yield`
                // keyword (see "generators" unstable feature).
                // FIXME(richkadel): Add tests with async and threading.

                | TerminatorKind::SwitchInt { .. } => {
                    // The `bb` has more than one _outgoing_ edge, or exits the function. Save the
                    // current sequence of `basic_blocks` gathered to this point, as a new
                    // `BasicCoverageBlockData`.
                    Self::add_basic_coverage_block(
                        &mut bcbs,
                        &mut bb_to_bcb,
                        basic_blocks.split_off(0),
                    );
                    // TODO(richkadel): uncomment debug!
                    // debug!("  because term.kind = {:?}", term.kind);
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
            // TODO(richkadel): uncomment debug!
            // debug!("  because the end of the MIR CFG was reached while traversing");
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
        // TODO(richkadel): uncomment debug!
        // debug!("adding bcb{}: {:?}", bcb.index(), bcb_data);
        bcbs.push(bcb_data);
    }

    #[inline(always)]
    pub fn iter_enumerated(
        &self,
    ) -> impl Iterator<Item = (BasicCoverageBlock, &BasicCoverageBlockData)> {
        self.bcbs.iter_enumerated()
    }

    #[inline(always)]
    pub fn bcb_from_bb(&self, bb: BasicBlock) -> BasicCoverageBlock {
        self.bb_to_bcb[bb]
            .expect("bb is not in any bcb (pre-filtered, such as unwind paths perhaps?)")
    }

    #[inline(always)]
    pub fn compute_bcb_dominators(&self) -> Dominators<BasicCoverageBlock> {
        dominators::dominators(self)
    }
}

impl Index<BasicCoverageBlock> for BasicCoverageBlocks {
    type Output = BasicCoverageBlockData;

    #[inline]
    fn index(&self, index: BasicCoverageBlock) -> &BasicCoverageBlockData {
        &self.bcbs[index]
    }
}

impl IndexMut<BasicCoverageBlock> for BasicCoverageBlocks {
    #[inline]
    fn index_mut(&mut self, index: BasicCoverageBlock) -> &mut BasicCoverageBlockData {
        &mut self.bcbs[index]
    }
}

impl graph::DirectedGraph for BasicCoverageBlocks {
    type Node = BasicCoverageBlock;
}

impl graph::WithNumNodes for BasicCoverageBlocks {
    #[inline]
    fn num_nodes(&self) -> usize {
        self.bcbs.len()
    }
}

impl graph::WithStartNode for BasicCoverageBlocks {
    #[inline]
    fn start_node(&self) -> Self::Node {
        self.bcb_from_bb(mir::START_BLOCK)
    }
}

type BcbSuccessors<'a> = std::slice::Iter<'a, BasicCoverageBlock>;

impl graph::WithSuccessors for BasicCoverageBlocks {
    #[inline]
    fn successors(&self, node: Self::Node) -> <Self as GraphSuccessors<'_>>::Iter {
        self.successors[node].iter().cloned()
    }
}

impl<'a> graph::GraphSuccessors<'a> for BasicCoverageBlocks {
    type Item = BasicCoverageBlock;
    type Iter = std::iter::Cloned<BcbSuccessors<'a>>;
}

// `BasicBlock` `Predecessors` uses a `SmallVec` of length 4 because, "Typically 95%+ of basic
// blocks have 4 or fewer predecessors." BasicCoverageBlocks should have the same or less.
type BcbPredecessors = SmallVec<[BasicCoverageBlock; 4]>;

impl graph::GraphPredecessors<'graph> for BasicCoverageBlocks {
    type Item = BasicCoverageBlock;
    type Iter = smallvec::IntoIter<[BasicCoverageBlock; 4]>;
}

impl graph::WithPredecessors for BasicCoverageBlocks {
    #[inline]
    fn predecessors(&self, node: Self::Node) -> <Self as graph::GraphPredecessors<'_>>::Iter {
        self.predecessors[node].clone().into_iter()
    }
}

#[derive(Debug, Copy, Clone)]
enum CoverageStatement {
    Statement(BasicBlock, Span, usize),
    Terminator(BasicBlock, Span),
}

impl CoverageStatement {
    pub fn format(&self, tcx: TyCtxt<'tcx>, mir_body: &'a mir::Body<'tcx>) -> String {
        match *self {
            Self::Statement(bb, span, stmt_index) => {
                let stmt = &mir_body[bb].statements[stmt_index];
                format!(
                    "{}: @{}[{}]: {:?}",
                    spanview::source_range_no_file(tcx, &span),
                    bb.index(),
                    stmt_index,
                    stmt
                )
            }
            Self::Terminator(bb, span) => {
                let term = mir_body[bb].terminator();
                format!(
                    "{}: @{}.{}: {:?}",
                    spanview::source_range_no_file(tcx, &span),
                    bb.index(),
                    term_type(&term.kind),
                    term.kind
                )
            }
        }
    }

    pub fn span(&self) -> &Span {
        match self {
            Self::Statement(_, span, _) | Self::Terminator(_, span) => span,
        }
    }
}

fn term_type(kind: &TerminatorKind<'tcx>) -> &'static str {
    match kind {
        TerminatorKind::Goto { .. } => "Goto",
        TerminatorKind::SwitchInt { .. } => "SwitchInt",
        TerminatorKind::Resume => "Resume",
        TerminatorKind::Abort => "Abort",
        TerminatorKind::Return => "Return",
        TerminatorKind::Unreachable => "Unreachable",
        TerminatorKind::Drop { .. } => "Drop",
        TerminatorKind::DropAndReplace { .. } => "DropAndReplace",
        TerminatorKind::Call { .. } => "Call",
        TerminatorKind::Assert { .. } => "Assert",
        TerminatorKind::Yield { .. } => "Yield",
        TerminatorKind::GeneratorDrop => "GeneratorDrop",
        TerminatorKind::FalseEdge { .. } => "FalseEdge",
        TerminatorKind::FalseUnwind { .. } => "FalseUnwind",
        TerminatorKind::InlineAsm { .. } => "InlineAsm",
    }
}

/// A BCB is deconstructed into one or more `Span`s. Each `Span` maps to a `CoverageSpan` that
/// references the originating BCB and one or more MIR `Statement`s and/or `Terminator`s.
/// Initially, the `Span`s come from the `Statement`s and `Terminator`s, but subsequent
/// transforms can combine adjacent `Span`s and `CoverageSpan` from the same BCB, merging the
/// `CoverageStatement` vectors, and the `Span`s to cover the extent of the combined `Span`s.
///
/// Note: A `CoverageStatement` merged into another CoverageSpan may come from a `BasicBlock` that
/// is not part of the `CoverageSpan` bcb if the statement was included because it's `Span` matches
/// or is subsumed by the `Span` associated with this `CoverageSpan`, and it's `BasicBlock`
/// `is_dominated_by()` the `BasicBlock`s in this `CoverageSpan`.
#[derive(Debug, Clone)]
struct CoverageSpan {
    span: Span,
    bcb: BasicCoverageBlock,
    coverage_statements: Vec<CoverageStatement>,
    is_closure: bool,
}

impl CoverageSpan {
    pub fn for_statement(
        statement: &Statement<'tcx>,
        span: Span,
        bcb: BasicCoverageBlock,
        bb: BasicBlock,
        stmt_index: usize,
    ) -> Self {
        let is_closure = match statement.kind {
            StatementKind::Assign(box (
                _,
                Rvalue::Aggregate(box AggregateKind::Closure(_, _), _),
            )) => true,
            _ => false,
        };

        Self {
            span,
            bcb,
            coverage_statements: vec![CoverageStatement::Statement(bb, span, stmt_index)],
            is_closure,
        }
    }

    pub fn for_terminator(span: Span, bcb: BasicCoverageBlock, bb: BasicBlock) -> Self {
        Self {
            span,
            bcb,
            coverage_statements: vec![CoverageStatement::Terminator(bb, span)],
            is_closure: false,
        }
    }

    pub fn merge_from(&mut self, mut other: CoverageSpan) {
        debug_assert!(self.is_mergeable(&other));
        self.span = self.span.to(other.span);
        if other.is_closure {
            self.is_closure = true;
        }
        self.coverage_statements.append(&mut other.coverage_statements);
    }

    pub fn cutoff_statements_at(&mut self, cutoff_pos: BytePos) {
        self.coverage_statements.retain(|covstmt| covstmt.span().hi() <= cutoff_pos);
        if let Some(highest_covstmt) =
            self.coverage_statements.iter().max_by_key(|covstmt| covstmt.span().hi())
        {
            self.span = self.span.with_hi(highest_covstmt.span().hi());
        }
    }

    #[inline]
    pub fn is_dominated_by(
        &self,
        other: &CoverageSpan,
        bcb_dominators: &Dominators<BasicCoverageBlock>,
    ) -> bool {
        debug_assert!(!self.is_in_same_bcb(other));
        bcb_dominators.is_dominated_by(self.bcb, other.bcb)
    }

    #[inline]
    pub fn is_mergeable(&self, other: &Self) -> bool {
        self.is_in_same_bcb(other) && !(self.is_closure || other.is_closure)
    }

    #[inline]
    pub fn is_in_same_bcb(&self, other: &Self) -> bool {
        self.bcb == other.bcb
    }
}

/// Maintains separate worklists for each loop in the BasicCoverageBlock CFG, plus one for the
/// BasicCoverageBlocks outside all loops. This supports traversing the BCB CFG in a way that
/// ensures a loop is completely traversed before processing Blocks after the end of the loop.
#[derive(Debug)]
struct TraversalContext {
    /// the start (backedge target) of a loop. If `None`, the context is all
    /// BasicCoverageBlocks in the MIR that are _not_ within any loop.
    loop_header: Option<BasicCoverageBlock>,

    /// worklist, to be traversed, of BasicCoverageBlocks in the loop headed by
    /// `loop_header`, such that the loop is the inner inner-most loop containing these
    /// BasicCoverageBlocks
    worklist: Vec<BasicCoverageBlock>,
}

struct Instrumentor<'a, 'tcx> {
    pass_name: &'a str,
    tcx: TyCtxt<'tcx>,
    mir_body: &'a mut mir::Body<'tcx>,
    hir_body: &'tcx rustc_hir::Body<'tcx>,
    bcb_dominators: Dominators<BasicCoverageBlock>,
    basic_coverage_blocks: BasicCoverageBlocks,
    function_source_hash: Option<u64>,
    next_counter_id: u32,
    num_expressions: u32,
}

impl<'a, 'tcx> Instrumentor<'a, 'tcx> {
    fn new(pass_name: &'a str, tcx: TyCtxt<'tcx>, mir_body: &'a mut mir::Body<'tcx>) -> Self {
        let hir_body = hir_body(tcx, mir_body.source.def_id());
        let basic_coverage_blocks = BasicCoverageBlocks::from_mir(mir_body);
        let bcb_dominators = basic_coverage_blocks.compute_bcb_dominators();
        Self {
            pass_name,
            tcx,
            mir_body,
            hir_body,
            basic_coverage_blocks,
            bcb_dominators,
            function_source_hash: None,
            next_counter_id: CounterValueReference::START.as_u32(),
            num_expressions: 0,
        }
    }

    /// Counter IDs start from one and go up.
    fn next_counter(&mut self) -> CounterValueReference {
        assert!(self.next_counter_id < u32::MAX - self.num_expressions);
        let next = self.next_counter_id;
        self.next_counter_id += 1;
        CounterValueReference::from(next)
    }

    /// Expression IDs start from u32::MAX and go down because a CounterExpression can reference
    /// (add or subtract counts) of both Counter regions and CounterExpression regions. The counter
    /// expression operand IDs must be unique across both types.
    fn next_expression(&mut self) -> InjectedExpressionIndex {
        assert!(self.next_counter_id < u32::MAX - self.num_expressions);
        let next = u32::MAX - self.num_expressions;
        self.num_expressions += 1;
        InjectedExpressionIndex::from(next)
    }

    fn function_source_hash(&mut self) -> u64 {
        match self.function_source_hash {
            Some(hash) => hash,
            None => {
                let hash = hash_mir_source(self.tcx, self.hir_body);
                self.function_source_hash.replace(hash);
                hash
            }
        }
    }

    fn inject_counters(&'a mut self) {
        let tcx = self.tcx;
        let source_map = tcx.sess.source_map();
        let mir_source = self.mir_body.source;
        let def_id = mir_source.def_id();
        let body_span = self.body_span();
        let source_file = source_map.lookup_source_file(body_span.lo());
        let file_name = Symbol::intern(&source_file.name.to_string());

        debug!("instrumenting {:?}, span: {}", def_id, source_map.span_to_string(body_span));

        let coverage_spans = self.coverage_spans();

        let span_viewables = if pretty::dump_enabled(tcx, self.pass_name, def_id) {
            Some(self.span_viewables(&coverage_spans))
        } else {
            None
        };

        self.make_counters();

        // Inject a counter for each `CoverageSpan`. There can be multiple `CoverageSpan`s for a
        // given BCB, but only one actual counter needs to be incremented per BCB. `bb_counters`
        // maps each `bcb` to its `Counter`, when injected. Subsequent `CoverageSpan`s
        // for a BCB that already has a `Counter` will inject a `CounterExpression` instead, and
        // compute its value by adding `ZERO` to the BCB `Counter` value.
        let mut bcb_counters = IndexVec::from_elem_n(None, self.basic_coverage_blocks.num_nodes());
        for CoverageSpan { span, bcb, .. } in coverage_spans {
            let counter_kind = if let Some(&counter_operand) = bcb_counters[bcb].as_ref() {
                self.make_expression(counter_operand, Op::Add, ExpressionOperandId::ZERO)
            } else if let Some(counter_kind) = self.bcb_data_mut(bcb).take_counter() {
                bcb_counters[bcb] = Some(counter_kind.as_operand_id());
                counter_kind
            } else {
                bug!("Every BasicCoverageBlock should have a Counter or CounterExpression");
            };
            // TODO(richkadel): uncomment debug!
            // debug!(
            //     "Injecting {:?} at: {:?}:\n{}\n==========",
            //     counter_kind,
            //     span,
            //     source_map.span_to_snippet(span).expect("Error getting source for span"),
            // );
            self.inject_statement(file_name, &source_file, counter_kind, span, bcb);
        }

        if let Some(span_viewables) = span_viewables {
            let mut file =
                pretty::create_dump_file(tcx, "html", None, self.pass_name, &0, mir_source)
                    .expect("Unexpected error creating MIR spanview HTML file");
            let crate_name = tcx.crate_name(def_id.krate);
            let item_name = tcx.def_path(def_id).to_filename_friendly_no_crate();
            let title = format!("{}.{} - Coverage Spans", crate_name, item_name);
            spanview::write_document(tcx, def_id, span_viewables, &title, &mut file)
                .expect("Unexpected IO error dumping coverage spans as HTML");
        }
    }

    /// Traverse the BCB CFG and add either a `Counter` or `CounterExpression` to ever BCB, to be
    /// injected with `CoverageSpan`s. `CounterExpressions` have no runtime overhead, so if a viable
    /// expression (adding or subtracting two other counters or expressions) can compute the same
    /// result as an embedded counter, a `CounterExpression` should be used.
    ///
    /// If two `BasicCoverageBlocks` branch from another `BasicCoverageBlock`, one of the branches
    /// can be counted by `CounterExpression` by subtracting the other branch from the branching
    /// block. Otherwise, the `BasicCoverageBlock` executed the least should have the `Counter`.
    /// One way to predict which branch executes the least is by considering loops. A loop is exited
    /// at a branch, so the branch that jumps to a `BasicCoverageBlock` outside the loop is almost
    /// always executed less than the branch that does not exit the loop.
    fn make_counters(&mut self) {
        debug!(
            "make_counters(): adding a counter or expression to each BasicCoverageBlock.\n    ... First identify any loops by their backedges:"
        );
        let mut loop_headers = BitSet::new_empty(self.basic_coverage_blocks.num_nodes());

        // Identify backedges
        for (bcb, _) in self.basic_coverage_blocks.iter_enumerated() {
            for &successor in &self.basic_coverage_blocks.successors[bcb] {
                if self.bcb_is_dominated_by(bcb, successor) {
                    debug!("Found BCB backedge: {:?} -> loop_header: {:?}", bcb, successor);
                    loop_headers.insert(successor);
                }
            }
        }

        let start_bcb = self.basic_coverage_blocks.start_node();

        // `context_stack` starts with a `TraversalContext` for the main function context (beginning
        // with the `start` BasicCoverageBlock of the function). New worklists are pushed to the top
        // of the stack as loops are entered, and popped off of the stack when a loop's worklist is
        // exhausted.
        let mut context_stack = Vec::new();
        context_stack.push(TraversalContext { loop_header: None, worklist: vec![start_bcb] });
        let mut visited = BitSet::new_empty(self.basic_coverage_blocks.num_nodes());

        while let Some(bcb) = {
            // Strip contexts with empty worklists from the top of the stack
            while context_stack
                .last()
                .map_or(false, |context| context.worklist.is_empty())
            {
                context_stack.pop();
            }
            context_stack.last_mut().map_or(None, |context| context.worklist.pop())
        }
        {
            if !visited.insert(bcb) {
                debug!("Already visited: {:?}", bcb);
                continue;
            }
            debug!("Visiting {:?}", bcb);
            if loop_headers.contains(bcb) {
                debug!("{:?} is a loop header! Start a new TraversalContext...", bcb);
                context_stack
                    .push(TraversalContext { loop_header: Some(bcb), worklist: Vec::new() });
            }

            debug!(
                "{:?} has {} successors:",
                bcb,
                self.basic_coverage_blocks.successors[bcb].len()
            );
            for &successor in &self.basic_coverage_blocks.successors[bcb] {
                for context in context_stack.iter_mut().rev() {
                    if let Some(loop_header) = context.loop_header {
                        if self.bcb_is_dominated_by(successor, loop_header) {
                            debug!(
                                "Adding successor {:?} to worklist of loop headed by {:?}",
                                successor, loop_header
                            );
                            context.worklist.push(successor);
                            break;
                        }
                    } else {
                        debug!("Adding successor {:?} to non-loop worklist", successor);
                        context.worklist.push(successor);
                    }
                }
            }

            let bcb_counter_operand = if let Some(counter) = self.bcb_data(bcb).counter() {
                debug!("{:?} already has a counter: {:?}", bcb, counter);
                counter.as_operand_id()
            } else {
                let counter = self.make_counter();
                debug!("{:?} needs a counter: {:?}", bcb, counter);
                let operand = counter.as_operand_id();
                self.bcb_data_mut(bcb).set_counter(counter);
                operand
            };

            let targets = match &self.bcb_data(bcb).terminator(self.mir_body) {
                TerminatorKind::SwitchInt { targets, .. } => targets.clone(),
                _ => vec![],
            };
            if targets.len() > 0 {
                debug!(
                    "{:?}'s terminator is a SwitchInt with targets: {:?}",
                    bcb,
                    targets.iter().map(|bb| self.bcb_from_bb(*bb)).collect::<Vec<_>>()
                );
                let switch_int_counter_operand = bcb_counter_operand;

                // Only one target can have an expression, but if `found_loop_exit`, any
                // `in_loop_target` can get the `CounterExpression`.
                let mut some_in_loop_target = None;
                for context in context_stack.iter().rev() {
                    if let Some(loop_header) = context.loop_header {
                        let mut found_loop_exit = false;
                        for &bb in &targets {
                            let target_bcb = self.bcb_from_bb(bb);
// TODO(richkadel): But IF...
                            if self.bcb_is_dominated_by(target_bcb, loop_header) {
//                  the target or any non-branching BCB successor down the line
//                  exits or is a TerminatorKind::Return, or something similar,
//                  then this should be "found_loop_exit" instead of some_in_loop_target


// WHAT IF instead of checking target_bcb is dominated by loop header, 
// we check something like, if backedge start (for all backedges leading to loop header?)
// is dominated by target_bcb?
// AND CAN THERE BE MORE THAN ONE BACKEDGE?  I THINK MAYBE... like "continue loop_label;"

// Will that work better?
// YES I THINK SO... IT SAYS, "target_bcb" leads out of the loop or it doesn't.

                                some_in_loop_target = Some(target_bcb);
                            } else {
                                found_loop_exit = true;
                            }
                            if some_in_loop_target.is_some() && found_loop_exit {
                                break;
                            }
                        }
                        debug!(
                            "found_loop_exit={}, some_in_loop_target={:?}",
                            found_loop_exit, some_in_loop_target
                        );
                        if !(found_loop_exit && some_in_loop_target.is_none()) {
                            break;
                        }
                        // else all branches exited this loop context, so run the same checks with
                        // the outer loop(s)
                    }
                }

                // If some preferred target for a CounterExpression was not determined, pick any
                // target.
                let expression_target = if let Some(in_loop_target) = some_in_loop_target {
                    debug!("Adding expression to in_loop_target={:?}", some_in_loop_target);
                    in_loop_target
                } else {
                    let bb_without_counter = *targets
                        .iter()
                        .find(|&&bb| {
                            let target_bcb = self.bcb_from_bb(bb);
                            self.bcb_data_mut(target_bcb).counter().is_none()
                        })
                        .expect("At least one target should need a counter");
                    debug!(
                        "No preferred expression target, so adding expression to the first target without an existing counter={:?}",
                        self.bcb_from_bb(bb_without_counter)
                    );
                    self.bcb_from_bb(bb_without_counter)
                };

                // Assign a Counter or CounterExpression to each target BasicCoverageBlock,
                // computing intermediate CounterExpression as needed.
                let mut some_prev_counter_operand = None;
                for bb in targets {
                    let target_bcb = self.bcb_from_bb(bb);
                    if target_bcb != expression_target {
                        // TODO(richkadel): this let if let else block is repeated above. Refactor into function.
                        let target_counter_operand =
                            if let Some(counter) = self.bcb_data(target_bcb).counter() {
                                debug!("{:?} already has a counter: {:?}", target_bcb, counter);
                                counter.as_operand_id()
                            } else {
                                let counter = self.make_counter();
                                debug!("{:?} gets a counter: {:?}", target_bcb, counter);
                                let operand = counter.as_operand_id();
                                self.bcb_data_mut(target_bcb).set_counter(counter);
                                operand
                            };
                        if let Some(prev_counter_operand) =
                            some_prev_counter_operand.replace(target_counter_operand)
                        {
                            let expression = self.make_expression(
                                prev_counter_operand,
                                Op::Add,
                                target_counter_operand,
                            );
                            debug!("new non-code expression: {:?}", expression);
                            let expression_operand = expression.as_operand_id();
                            self.inject_non_code_expression(expression);
                            some_prev_counter_operand.replace(expression_operand);
                        }
                    }
                }
                let expression = self.make_expression(
                    switch_int_counter_operand,
                    Op::Subtract,
                    some_prev_counter_operand.expect("prev_counter should have a value"),
                );
                debug!("{:?} gets an expression: {:?}", expression_target, expression);
                self.bcb_data_mut(expression_target).set_counter(expression);
            }
        }

        debug_assert_eq!(visited.count(), visited.domain_size());
    }

    #[inline]
    fn bcb_from_bb(&self, bb: BasicBlock) -> BasicCoverageBlock {
        self.basic_coverage_blocks.bcb_from_bb(bb)
    }

    #[inline]
    fn bcb_data(&self, bcb: BasicCoverageBlock) -> &BasicCoverageBlockData {
        &self.basic_coverage_blocks[bcb]
    }

    #[inline]
    fn bcb_data_mut(&mut self, bcb: BasicCoverageBlock) -> &mut BasicCoverageBlockData {
        &mut self.basic_coverage_blocks[bcb]
    }

    #[inline]
    fn bcb_is_dominated_by(&self, node: BasicCoverageBlock, dom: BasicCoverageBlock) -> bool {
        self.bcb_dominators.is_dominated_by(node, dom)
    }

    // loop through backedges

    // select inner loops before their outer loops, so the first matched loop for a given target_bcb
    // is it's inner-most loop

    // CASE #1:
    // if a target_bcb is_dominated_by a loop bcb (target of backedge), and if any other target_bcb is NOT dominated by the loop bcb,
    // add expression to the first target_bcb that dominated by the loop bcb, and counters to all others. Compute expressions from
    // counter pairs as needed, to provide a single sum that can be subtracted from the SwitchInt block's counter.

    // CASE #2:
    // if all target_bcb are dominated_by the loop bcb, no branch ends the loop (directly), so pick any branch to have the expression,

    // CASE #3:
    // if NONE of the target_bcb are dominated_by the loop bcb, check if there's an outer loop (from stack of active loops?)
    // and re-do this check again to see if one of them jumps out of the outer loop while other(s) don't, and assign the expression
    // to one of the target_bcb that is dominated_by that outer loop. (Continue this if none are dominated by the outer loop either.)

    // TODO(richkadel): In the last case above, also see the next TODO below. If all targets exit the loop then can we pass that info
    // to the predecessor (if only one??) so if the predecessor is a target of another SwitchInt, we know that the predecessor exits
    // the loop, and should have the counter, if the predecessor is in CASE #2 (none of the other targets of the predecessor's
    // SwitchInt exited the loop?)

    // TODO(richkadel): What about a target that is another SwitchInt where both branches exit the loop?
    // Can I detect that somehow?

    // TODO(richkadel): For any node, N, and one of its successors, H (so N -> H), if (_also_)
    // N is_dominated_by H, then N -> H is a backedge. That is, we've identified that N -> H is
    // at least _one_ of possibly multiple arcs that loop back to the start of the loop with
    // "header" H, and this also means we've identified a loop, that has "header" H.
    //
    // H dominates everything inside the loop.
    //
    // So a SwitchInt target in a BasicBlock that is_dominated_by H and has a branch target to a
    // BasicBlock that is:
    //   * not H, and   ... (what if the SwitchInt branch target _is_ H? `continue`? is this a
    //     candidate for a middle or optional priority for getting a Counter?)
    //   * not is_dominated_by H
    // is a branch that jumps outside the loop, and should get an actual Counter, most likely
    //
    // Or perhaps conversely, a SwitchInt dominated by H with a branch that has a target that
    // ALSO is dominated by H should get a CounterExpression.
    //
    //
    // So I need to identify all of the "H"'s, by identifying all of the backedges.
    //
    // If I have multiple H's (multiple loops), how do I decide which loop to compare a branch
    // target (by dominator) to?
    //
    // Can I assume the traversal order is helpful here? I.e., the "last" encountered loop
    // header is the (only?) one to compare to? (Probably not only... I don't see how that would
    // work for nested loops.)
    //
    // What about multiple loops in sequence?
    //
    //
    // What about nexted loops and jumping out of one or more of them at a time?

    fn make_counter(&mut self) -> CoverageKind {
        CoverageKind::Counter {
            function_source_hash: self.function_source_hash(),
            id: self.next_counter(),
        }
    }

    fn make_expression(
        &mut self,
        lhs: ExpressionOperandId,
        op: Op,
        rhs: ExpressionOperandId,
    ) -> CoverageKind {
        CoverageKind::Expression { id: self.next_expression(), lhs, op, rhs }
    }

    fn inject_statement(
        &mut self,
        file_name: Symbol,
        source_file: &Lrc<SourceFile>,
        counter_kind: CoverageKind,
        span: Span,
        bcb: BasicCoverageBlock,
    ) {
        let code_region = Some(make_code_region(file_name, source_file, span));
        // TODO(richkadel): uncomment debug!
        // debug!("  injecting statement {:?} covering {:?}", counter_kind, code_region);

        let inject_in_bb = self.bcb_data(bcb).leader_bb();
        let data = &mut self.mir_body[inject_in_bb];
        let source_info = data.terminator().source_info;
        let statement = Statement {
            source_info,
            kind: StatementKind::Coverage(box Coverage { kind: counter_kind, code_region }),
        };
        data.statements.push(statement);
    }

    // Non-code expressions are injected into the coverage map, without generating executable code.
    fn inject_non_code_expression(&mut self, expression: CoverageKind) {
        debug_assert!(if let CoverageKind::Expression { .. } = expression { true } else { false });
        // TODO(richkadel): uncomment debug!
        // debug!("  injecting non-code expression {:?}", expression);

        let inject_in_bb = mir::START_BLOCK;
        let data = &mut self.mir_body[inject_in_bb];
        let source_info = data.terminator().source_info;
        let statement = Statement {
            source_info,
            kind: StatementKind::Coverage(box Coverage { kind: expression, code_region: None }),
        };
        data.statements.push(statement);
    }

    /// Converts the computed `BasicCoverageBlockData`s into `SpanViewable`s.
    fn span_viewables(&self, coverage_spans: &Vec<CoverageSpan>) -> Vec<SpanViewable> {
        let tcx = self.tcx;
        let mut span_viewables = Vec::new();
        for coverage_span in coverage_spans {
            let CoverageSpan { span, bcb, coverage_statements, .. } = coverage_span;
            let bcb_data = self.bcb_data(*bcb);
            let id = bcb_data.id();
            let leader_bb = bcb_data.leader_bb();
            let mut sorted_coverage_statements = coverage_statements.clone();
            sorted_coverage_statements.sort_unstable_by_key(|covstmt| match *covstmt {
                CoverageStatement::Statement(bb, _, index) => (bb, index),
                CoverageStatement::Terminator(bb, _) => (bb, usize::MAX),
            });
            let tooltip = sorted_coverage_statements
                .iter()
                .map(|covstmt| covstmt.format(tcx, self.mir_body))
                .collect::<Vec<_>>()
                .join("\n");
            span_viewables.push(SpanViewable { bb: leader_bb, span: *span, id, tooltip });
        }
        span_viewables
    }

    #[inline(always)]
    fn body_span(&self) -> Span {
        self.hir_body.value.span
    }

    // Generate a set of `CoverageSpan`s from the filtered set of `Statement`s and `Terminator`s of
    // the `BasicBlock`(s) in the given `BasicCoverageBlockData`. One `CoverageSpan` is generated
    // for each `Statement` and `Terminator`. (Note that subsequent stages of coverage analysis will
    // merge some `CoverageSpan`s, at which point a `CoverageSpan` may represent multiple
    // `Statement`s and/or `Terminator`s.)
    fn extract_spans(
        &self,
        bcb: BasicCoverageBlock,
        bcb_data: &'a BasicCoverageBlockData,
    ) -> Vec<CoverageSpan> {
        let body_span = self.body_span();
        bcb_data
            .basic_blocks()
            .map(|bbref| {
                let bb = *bbref;
                let data = &self.mir_body[bb];
                data.statements
                    .iter()
                    .enumerate()
                    .filter_map(move |(index, statement)| {
                        filtered_statement_span(statement, body_span).map(|span| {
                            CoverageSpan::for_statement(statement, span, bcb, bb, index)
                        })
                    })
                    .chain(
                        filtered_terminator_span(data.terminator(), body_span)
                            .map(|span| CoverageSpan::for_terminator(span, bcb, bb)),
                    )
            })
            .flatten()
            .collect()
    }

    /// Generate a minimal set of `CoverageSpan`s, each representing a contiguous code region to be
    /// counted.
    ///
    /// The basic steps are:
    ///
    /// 1. Extract an initial set of spans from the `Statement`s and `Terminator`s of each
    ///    `BasicCoverageBlockData`.
    /// 2. Sort the spans by span.lo() (starting position). Spans that start at the same position
    ///    are sorted with longer spans before shorter spans; and equal spans are sorted
    ///    (deterministically) based on "dominator" relationship (if any).
    /// 3. Traverse the spans in sorted order to identify spans that can be dropped (for instance,
    ///    if another span or spans are already counting the same code region), or should be merged
    ///    into a broader combined span (because it represents a contiguous, non-branching, and
    ///    uninterrupted region of source code).
    ///
    ///    Closures are exposed in their enclosing functions as `Assign` `Rvalue`s, and since
    ///    closures have their own MIR, their `Span` in their enclosing function should be left
    ///    "uncovered".
    ///
    /// Note the resulting vector of `CoverageSpan`s does may not be fully sorted (and does not need
    /// to be).
    fn coverage_spans(&self) -> Vec<CoverageSpan> {
        let mut initial_spans = Vec::<CoverageSpan>::with_capacity(self.mir_body.num_nodes() * 2);
        for (bcb, bcb_data) in self.basic_coverage_blocks.iter_enumerated() {
            for coverage_span in self.extract_spans(bcb, bcb_data) {
                initial_spans.push(coverage_span);
            }
        }

        if initial_spans.is_empty() {
            // This can happen if, for example, the function is unreachable (contains only a
            // `BasicBlock`(s) with an `Unreachable` terminator).
            return initial_spans;
        }

        initial_spans.sort_unstable_by(|a, b| {
            if a.span.lo() == b.span.lo() {
                if a.span.hi() == b.span.hi() {
                    if a.is_in_same_bcb(b) {
                        Some(Ordering::Equal)
                    } else {
                        // Sort equal spans by dominator relationship, in reverse order (so
                        // dominators always come after the dominated equal spans). When later
                        // comparing two spans in order, the first will either dominate the second,
                        // or they will have no dominator relationship.
                        self.bcb_dominators.rank_partial_cmp(b.bcb, a.bcb)
                    }
                } else {
                    // Sort hi() in reverse order so shorter spans are attempted after longer spans.
                    // This guarantees that, if a `prev` span overlaps, and is not equal to, a
                    // `curr` span, the prev span either extends further left of the curr span, or
                    // they start at the same position and the prev span extends further right of
                    // the end of the curr span.
                    b.span.hi().partial_cmp(&a.span.hi())
                }
            } else {
                a.span.lo().partial_cmp(&b.span.lo())
            }
            .unwrap()
        });

        let refinery = CoverageSpanRefinery::from_sorted_spans(initial_spans, &self.bcb_dominators);
        refinery.to_refined_spans()
    }
}

struct CoverageSpanRefinery<'a> {
    sorted_spans_iter: std::vec::IntoIter<CoverageSpan>,
    bcb_dominators: &'a Dominators<BasicCoverageBlock>,
    some_curr: Option<CoverageSpan>,
    curr_original_span: Span,
    some_prev: Option<CoverageSpan>,
    prev_original_span: Span,
    pending_dups: Vec<CoverageSpan>,
    refined_spans: Vec<CoverageSpan>,
}

impl<'a> CoverageSpanRefinery<'a> {
    fn from_sorted_spans(
        sorted_spans: Vec<CoverageSpan>,
        bcb_dominators: &'a Dominators<BasicCoverageBlock>,
    ) -> Self {
        let refined_spans = Vec::with_capacity(sorted_spans.len());
        let mut sorted_spans_iter = sorted_spans.into_iter();
        let prev = sorted_spans_iter.next().expect("at least one span");
        let prev_original_span = prev.span;
        Self {
            sorted_spans_iter,
            bcb_dominators,
            refined_spans,
            some_curr: None,
            curr_original_span: Span::with_root_ctxt(BytePos(0), BytePos(0)),
            some_prev: Some(prev),
            prev_original_span,
            pending_dups: Vec::new(),
        }
    }

    /// Iterate through the sorted `CoverageSpan`s, and return the refined list of merged and
    /// de-duplicated `CoverageSpan`s.
    fn to_refined_spans(mut self) -> Vec<CoverageSpan> {
        while self.next_coverage_span() {
            if self.curr().is_mergeable(self.prev()) {
                // TODO(richkadel): uncomment debug!
                // debug!("  same bcb (and neither is a closure), merge with prev={:?}", self.prev());
                let prev = self.take_prev();
                self.curr_mut().merge_from(prev);
            // Note that curr.span may now differ from curr_original_span
            } else if self.prev_ends_before_curr() {
                // TODO(richkadel): uncomment debug!
                // debug!(
                //     "  different bcbs and disjoint spans, so keep curr for next iter, and add \
                //     prev={:?}",
                //     self.prev()
                // );
                let prev = self.take_prev();
                self.add_refined_span(prev);
            } else if self.prev().is_closure {
                // drop any equal or overlapping span (`curr`) and keep `prev` to test again in the
                // next iter
                // TODO(richkadel): uncomment debug!
                // debug!(
                //     "  curr overlaps a closure (prev). Drop curr and keep prev for next iter. \
                //     prev={:?}",
                //     self.prev()
                // );
                self.discard_curr();
            } else if self.curr().is_closure {
                self.carve_out_span_for_closure();
            } else if self.prev_original_span == self.curr().span {
                self.hold_pending_dups_unless_dominated();
            } else {
                self.cutoff_prev_at_overlapping_curr();
            }
        }
        // TODO(richkadel): uncomment debug!
        // debug!("    AT END, adding last prev={:?}", self.prev());
        let pending_dups = self.pending_dups.split_off(0);
        for dup in pending_dups.into_iter() {
            // TODO(richkadel): uncomment debug!
            // debug!("    ...adding at least one pending dup={:?}", dup);
            self.add_refined_span(dup);
        }
        let prev = self.take_prev();
        self.add_refined_span(prev);

        // FIXME(richkadel): Replace some counters with expressions if they can be calculated based
        // on branching. (For example, one branch of a SwitchInt can be computed from the counter
        // for the CoverageSpan just prior to the SwitchInt minus the sum of the counters of all
        // other branches).

        self.to_refined_spans_without_closures()
    }

    fn add_refined_span(&mut self, coverage_span: CoverageSpan) {
        self.refined_spans.push(coverage_span);
    }

    /// Remove `CoverageSpan`s derived from closures, originally added to ensure the coverage
    /// regions for the current function leave room for the closure's own coverage regions
    /// (injected separately, from the closure's own MIR).
    fn to_refined_spans_without_closures(mut self) -> Vec<CoverageSpan> {
        self.refined_spans.retain(|covspan| !covspan.is_closure);
        self.refined_spans
    }

    fn curr(&self) -> &CoverageSpan {
        self.some_curr
            .as_ref()
            .unwrap_or_else(|| bug!("invalid attempt to unwrap a None some_curr"))
    }

    fn curr_mut(&mut self) -> &mut CoverageSpan {
        self.some_curr
            .as_mut()
            .unwrap_or_else(|| bug!("invalid attempt to unwrap a None some_curr"))
    }

    fn prev(&self) -> &CoverageSpan {
        self.some_prev
            .as_ref()
            .unwrap_or_else(|| bug!("invalid attempt to unwrap a None some_prev"))
    }

    fn prev_mut(&mut self) -> &mut CoverageSpan {
        self.some_prev
            .as_mut()
            .unwrap_or_else(|| bug!("invalid attempt to unwrap a None some_prev"))
    }

    fn take_prev(&mut self) -> CoverageSpan {
        self.some_prev.take().unwrap_or_else(|| bug!("invalid attempt to unwrap a None some_prev"))
    }

    /// If there are `pending_dups` but `prev` is not a matching dup (`prev.span` doesn't match the
    /// `pending_dups` spans), then one of the following two things happened during the previous
    /// iteration:
    ///   * the `span` of prev was modified (by `curr_mut().merge_from(prev)`); or
    ///   * the `span` of prev advanced past the end of the span of pending_dups
    ///     (`prev().span.hi() <= curr().span.lo()`)
    /// In either case, no more spans will match the span of `pending_dups`, so
    /// add the `pending_dups` if they don't overlap `curr`, and clear the list.
    fn check_pending_dups(&mut self) {
        if let Some(dup) = self.pending_dups.last() {
            if dup.span != self.prev().span {
                // TODO(richkadel): uncomment debug!
                // debug!(
                //     "    SAME spans, but pending_dups are NOT THE SAME, so BCBs matched on \
                //     previous iteration, or prev started a new disjoint span"
                // );
                if dup.span.hi() <= self.curr().span.lo() {
                    let pending_dups = self.pending_dups.split_off(0);
                    for dup in pending_dups.into_iter() {
                        // TODO(richkadel): uncomment debug!
                        // debug!("    ...adding at least one pending={:?}", dup);
                        self.add_refined_span(dup);
                    }
                } else {
                    self.pending_dups.clear();
                }
            }
        }
    }

    /// Advance `prev` to `curr` (if any), and `curr` to the next `CoverageSpan` in sorted order.
    fn next_coverage_span(&mut self) -> bool {
        if let Some(curr) = self.some_curr.take() {
            self.some_prev = Some(curr);
            self.prev_original_span = self.curr_original_span;
        }
        while let Some(curr) = self.sorted_spans_iter.next() {
            // TODO(richkadel): uncomment debug!
            // debug!("FOR curr={:?}", curr);
            if self.prev_starts_after_next(&curr) {
                // TODO(richkadel): uncomment debug!
                // debug!(
                //     "  prev.span starts after curr.span, so curr will be dropped (skipping past \
                //     closure?); prev={:?}",
                //     self.prev()
                // );
            } else {
                // Save a copy of the original span for `curr` in case the `CoverageSpan` is changed
                // by `self.curr_mut().merge_from(prev)`.
                self.curr_original_span = curr.span;
                self.some_curr.replace(curr);
                self.check_pending_dups();
                return true;
            }
        }
        false
    }

    /// If called, then the next call to `next_coverage_span()` will *not* update `prev` with the
    /// `curr` coverage span.
    fn discard_curr(&mut self) {
        self.some_curr = None;
    }

    /// Returns true if the curr span should be skipped because prev has already advanced beyond the
    /// end of curr. This can only happen if a prior iteration updated `prev` to skip past a region
    /// of code, such as skipping past a closure.
    fn prev_starts_after_next(&self, next_curr: &CoverageSpan) -> bool {
        self.prev().span.lo() > next_curr.span.lo()
    }

    /// Returns true if the curr span starts past the end of the prev span, which means they don't
    /// overlap, so we now know the prev can be added to the refined coverage spans.
    fn prev_ends_before_curr(&self) -> bool {
        self.prev().span.hi() <= self.curr().span.lo()
    }

    /// If `prev`s span extends left of the closure (`curr`), carve out the closure's
    /// span from `prev`'s span. (The closure's coverage counters will be injected when
    /// processing the closure's own MIR.) Add the portion of the span to the left of the
    /// closure; and if the span extends to the right of the closure, update `prev` to
    /// that portion of the span. For any `pending_dups`, repeat the same process.
    fn carve_out_span_for_closure(&mut self) {
        let curr_span = self.curr().span;
        let left_cutoff = curr_span.lo();
        let right_cutoff = curr_span.hi();
        let has_pre_closure_span = self.prev().span.lo() < right_cutoff;
        let has_post_closure_span = self.prev().span.hi() > right_cutoff;
        let mut pending_dups = self.pending_dups.split_off(0);
        if has_pre_closure_span {
            let mut pre_closure = self.prev().clone();
            pre_closure.span = pre_closure.span.with_hi(left_cutoff);
            // TODO(richkadel): uncomment debug!
            // debug!("  prev overlaps a closure. Adding span for pre_closure={:?}", pre_closure);
            if !pending_dups.is_empty() {
                for mut dup in pending_dups.iter().cloned() {
                    dup.span = dup.span.with_hi(left_cutoff);
                    // TODO(richkadel): uncomment debug!
                    // debug!("    ...and at least one pre_closure dup={:?}", dup);
                    self.add_refined_span(dup);
                }
            }
            self.add_refined_span(pre_closure);
        }
        if has_post_closure_span {
            // Update prev.span to start after the closure (and discard curr)
            self.prev_mut().span = self.prev().span.with_lo(right_cutoff);
            self.prev_original_span = self.prev().span;
            for dup in pending_dups.iter_mut() {
                dup.span = dup.span.with_lo(right_cutoff);
            }
            self.pending_dups.append(&mut pending_dups);
            self.discard_curr(); // since self.prev() was already updated
        } else {
            pending_dups.clear();
        }
    }

    /// When two `CoverageSpan`s have the same `Span`, dominated spans can be discarded; but if
    /// neither `CoverageSpan` dominates the other, both (or possibly more than two) are held,
    /// until their disposition is determined. In this latter case, the `prev` dup is moved into
    /// `pending_dups` so the new `curr` dup can be moved to `prev` for the next iteration.
    fn hold_pending_dups_unless_dominated(&mut self) {
        // equal coverage spans are ordered by dominators before dominated (if any)
        debug_assert!(!self.prev().is_dominated_by(self.curr(), self.bcb_dominators));

        if self.curr().is_dominated_by(&self.prev(), self.bcb_dominators) {
            // If one span dominates the other, assocate the span with the dominator only.
            //
            // For example:
            //     match somenum {
            //         x if x < 1 => { ... }
            //     }...
            // The span for the first `x` is referenced by both the pattern block (every
            // time it is evaluated) and the arm code (only when matched). The counter
            // will be applied only to the dominator block.
            //
            // The dominator's (`prev`) execution count may be higher than the dominated
            // block's execution count, so drop `curr`.
            // TODO(richkadel): uncomment debug!
            // debug!(
            //     "  different bcbs but SAME spans, and prev dominates curr. Drop curr and \
            //     keep prev for next iter. prev={:?}",
            //     self.prev()
            // );
            self.discard_curr();
        } else {
            // Save `prev` in `pending_dups`. (`curr` will become `prev` in the next iteration.)
            // If the `curr` CoverageSpan is later discarded, `pending_dups` can be discarded as
            // well; but if `curr` is added to refined_spans, the `pending_dups` will also be added.
            // TODO(richkadel): uncomment debug!
            // debug!(
            //     "  different bcbs but SAME spans, and neither dominates, so keep curr for \
            //     next iter, and, pending upcoming spans (unless overlapping) add prev={:?}",
            //     self.prev()
            // );
            let prev = self.take_prev();
            self.pending_dups.push(prev);
        }
    }

    /// `curr` overlaps `prev`. If `prev`s span extends left of `curr`s span, keep _only_
    /// statements that end before `curr.lo()` (if any), and add the portion of the
    /// combined span for those statements. Any other statements have overlapping spans
    /// that can be ignored because `curr` and/or other upcoming statements/spans inside
    /// the overlap area will produce their own counters. This disambiguation process
    /// avoids injecting multiple counters for overlapping spans, and the potential for
    /// double-counting.
    fn cutoff_prev_at_overlapping_curr(&mut self) {
        // TODO(richkadel): uncomment debug!
        // debug!(
        //     "  different bcbs, overlapping spans, so ignore/drop pending and only add prev \
        //     if it has statements that end before curr={:?}",
        //     self.prev()
        // );
        if self.pending_dups.is_empty() {
            let curr_span = self.curr().span;
            self.prev_mut().cutoff_statements_at(curr_span.lo());
            if self.prev().coverage_statements.is_empty() {
                // TODO(richkadel): uncomment debug!
                // debug!("  ... no non-overlapping statements to add");
            } else {
                // TODO(richkadel): uncomment debug!
                // debug!("  ... adding modified prev={:?}", self.prev());
                let prev = self.take_prev();
                self.add_refined_span(prev);
            }
        } else {
            // with `pending_dups`, `prev` cannot have any statements that don't overlap
            self.pending_dups.clear();
        }
    }
}

fn filtered_statement_span(statement: &'a Statement<'tcx>, body_span: Span) -> Option<Span> {
    match statement.kind {
        // These statements have spans that are often outside the scope of the executed source code
        // for their parent `BasicBlock`.
        StatementKind::StorageLive(_)
        | StatementKind::StorageDead(_)
        // Coverage should not be encountered, but don't inject coverage coverage
        | StatementKind::Coverage(_)
        // Ignore `Nop`s
        | StatementKind::Nop => None,

        // FIXME(richkadel): Look into a possible issue assigning the span to a
        // FakeReadCause::ForGuardBinding, in this example:
        //     match somenum {
        //         x if x < 1 => { ... }
        //     }...
        // The BasicBlock within the match arm code included one of these statements, but the span
        // for it covered the `1` in this source. The actual statements have nothing to do with that
        // source span:
        //     FakeRead(ForGuardBinding, _4);
        // where `_4` is:
        //     _4 = &_1; (at the span for the first `x`)
        // and `_1` is the `Place` for `somenum`.
        //
        // The arm code BasicBlock already has its own assignment for `x` itself, `_3 = 1`, and I've
        // decided it's reasonable for that span (even though outside the arm code) to be part of
        // the counted coverage of the arm code execution, but I can't justify including the literal
        // `1` in the arm code. I'm pretty sure that, if the `FakeRead(ForGuardBinding)` has a
        // purpose in codegen, it's probably in the right BasicBlock, but if so, the `Statement`s
        // `source_info.span` can't be right.
        //
        // Consider correcting the span assignment, assuming there is a better solution, and see if
        // the following pattern can be removed here:
        StatementKind::FakeRead(cause, _) if cause == FakeReadCause::ForGuardBinding => None,

        // Retain spans from all other statements
        StatementKind::FakeRead(_, _) // Not including `ForGuardBinding`
        | StatementKind::Assign(_)
        | StatementKind::SetDiscriminant { .. }
        | StatementKind::LlvmInlineAsm(_)
        | StatementKind::Retag(_, _)
        | StatementKind::AscribeUserType(_, _) => {
            Some(source_info_span(&statement.source_info, body_span))
        }
    }
}

fn filtered_terminator_span(terminator: &'a Terminator<'tcx>, body_span: Span) -> Option<Span> {
    match terminator.kind {
        // These terminators have spans that don't positively contribute to computing a reasonable
        // span of actually executed source code. (For example, SwitchInt terminators extracted from
        // an `if condition { block }` has a span that includes the executed block, if true,
        // but for coverage, the code region executed, up to *and* through the SwitchInt,
        // actually stops before the if's block.)
        TerminatorKind::Unreachable // Unreachable blocks are not connected to the MIR CFG
        | TerminatorKind::Assert { .. }
        | TerminatorKind::Drop { .. }
        | TerminatorKind::DropAndReplace { .. }
        | TerminatorKind::SwitchInt { .. }
        | TerminatorKind::Goto { .. }
        // For `FalseEdge`, only the `real` branch is taken, so it is similar to a `Goto`.
        | TerminatorKind::FalseEdge { .. } => None,

        // Retain spans from all other terminators
        TerminatorKind::Resume
        | TerminatorKind::Abort
        | TerminatorKind::Return
        | TerminatorKind::Call { .. }
        | TerminatorKind::Yield { .. }
        | TerminatorKind::GeneratorDrop
        | TerminatorKind::FalseUnwind { .. }
        | TerminatorKind::InlineAsm { .. } => {
            Some(source_info_span(&terminator.source_info, body_span))
        }
    }
}

#[inline(always)]
fn source_info_span(source_info: &SourceInfo, body_span: Span) -> Span {
    let span = original_sp(source_info.span, body_span).with_ctxt(SyntaxContext::root());
    if body_span.contains(span) { span } else { body_span }
}

/// Convert the Span into its file name, start line and column, and end line and column
fn make_code_region(file_name: Symbol, source_file: &Lrc<SourceFile>, span: Span) -> CodeRegion {
    let (start_line, mut start_col) = source_file.lookup_file_pos(span.lo());
    let (end_line, end_col) = if span.hi() == span.lo() {
        let (end_line, mut end_col) = (start_line, start_col);
        // Extend an empty span by one character so the region will be counted.
        let CharPos(char_pos) = start_col;
        if char_pos > 0 {
            start_col = CharPos(char_pos - 1);
        } else {
            end_col = CharPos(char_pos + 1);
        }
        (end_line, end_col)
    } else {
        source_file.lookup_file_pos(span.hi())
    };
    CodeRegion {
        file_name,
        start_line: start_line as u32,
        start_col: start_col.to_u32() + 1,
        end_line: end_line as u32,
        end_col: end_col.to_u32() + 1,
    }
}

fn hir_body<'tcx>(tcx: TyCtxt<'tcx>, def_id: DefId) -> &'tcx rustc_hir::Body<'tcx> {
    let hir_node = tcx.hir().get_if_local(def_id).expect("expected DefId is local");
    let fn_body_id = hir::map::associated_body(hir_node).expect("HIR node is a function with body");
    tcx.hir().body(fn_body_id)
}

fn hash_mir_source<'tcx>(tcx: TyCtxt<'tcx>, hir_body: &'tcx rustc_hir::Body<'tcx>) -> u64 {
    let mut hcx = tcx.create_no_span_stable_hashing_context();
    hash(&mut hcx, &hir_body.value).to_smaller_hash()
}

fn hash(
    hcx: &mut StableHashingContext<'tcx>,
    node: &impl HashStable<StableHashingContext<'tcx>>,
) -> Fingerprint {
    let mut stable_hasher = StableHasher::new();
    node.hash_stable(hcx, &mut stable_hasher);
    stable_hasher.finish()
}

pub struct ShortCircuitPreorder<
    'a,
    'tcx,
    F: Fn(&'tcx TerminatorKind<'tcx>) -> mir::Successors<'tcx>,
> {
    body: &'a mir::Body<'tcx>,
    visited: BitSet<BasicBlock>,
    worklist: Vec<BasicBlock>,
    filtered_successors: F,
}

impl<'a, 'tcx, F: Fn(&'tcx TerminatorKind<'tcx>) -> mir::Successors<'tcx>>
    ShortCircuitPreorder<'a, 'tcx, F>
{
    pub fn new(
        body: &'a mir::Body<'tcx>,
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

impl<'a: 'tcx, 'tcx, F: Fn(&'tcx TerminatorKind<'tcx>) -> mir::Successors<'tcx>> Iterator
    for ShortCircuitPreorder<'a, 'tcx, F>
{
    type Item = (BasicBlock, &'a BasicBlockData<'tcx>);

    fn next(&mut self) -> Option<(BasicBlock, &'a BasicBlockData<'tcx>)> {
        while let Some(idx) = self.worklist.pop() {
            if !self.visited.insert(idx) {
                continue;
            }

            let data = &self.body[idx];

            if let Some(ref term) = data.terminator {
                self.worklist.extend((self.filtered_successors)(&term.kind));
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

// NOTE: Regarding past efforts and revelations when trying to identify `Unreachable` coverage spans
// from the MIR:
//
// TerminatorKind::FalseEdge targets from SwitchInt don't appear to be helpful in identifying
// unreachable code. I did test the theory, but the following changes were not beneficial. (I
// assumed that replacing some constants with non-deterministic variables might effect which blocks
// were targeted by a `FalseEdge` `imaginary_target`. It did not.)
//
// Also note that, if there is a way to identify BasicBlocks that are part of the MIR CFG, but not
// actually reachable, here are some other things to consider:
//
// Injecting unreachable code regions will probably require computing the set difference between the
// basic blocks found without filtering out unreachable blocks, and the basic blocks found with a
// filter (similar to or as an extension of the `filter_unwind_paths` filter); then computing the
// `CoverageSpans` without the filter; and then injecting `Counter`s or `CounterExpression`s for
// blocks that are not unreachable, or injecting `Unreachable` code regions otherwise. This seems
// straightforward, but not trivial.
//
// Alternatively, we might instead want to leave the unreachable blocks in (bypass the filter here),
// and inject the counters. This will result in counter values of zero (0) for unreachable code
// (and, notably, the code will be displayed with a red background by `llvm-cov show`).
//
// ```rust
//     TerminatorKind::SwitchInt { .. } => {
//         let some_imaginary_target = successors.clone().find_map(|&successor| {
//             let term = mir_body[successor].terminator();
//             if let TerminatorKind::FalseEdge { imaginary_target, .. } = term.kind {
//                 if mir_body.predecessors()[imaginary_target].len() == 1 {
//                     return Some(imaginary_target);
//                 }
//             }
//             None
//         });
//         if let Some(imaginary_target) = some_imaginary_target {
//             box successors.filter(move |&&successor| successor != imaginary_target)
//         } else {
//             box successors
//         }
//     }
// ```
//
// Note this also required changing the closure signature for the `ShortCurcuitPreorder` to:
//
// ```rust
//     F: Fn(&'tcx TerminatorKind<'tcx>) -> Box<dyn Iterator<Item = &BasicBlock> + 'a>,
// ```
