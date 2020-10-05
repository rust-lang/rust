use crate::transform::MirPass;
use crate::util::pretty;
use crate::util::spanview::{self, source_range_no_file, SpanViewable};

use rustc_data_structures::fingerprint::Fingerprint;
use rustc_data_structures::graph::dominators::Dominators;
use rustc_data_structures::graph::WithNumNodes;
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
    Rvalue, Statement, StatementKind, Terminator, TerminatorKind,
};
use rustc_middle::ty::query::Providers;
use rustc_middle::ty::TyCtxt;
use rustc_span::def_id::DefId;
use rustc_span::source_map::original_sp;
use rustc_span::{BytePos, CharPos, Pos, SourceFile, Span, Symbol, SyntaxContext};

use std::cmp::Ordering;

const ID_SEPARATOR: &str = ",";

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
///
/// This visitor runs twice, first with `add_missing_operands` set to `false`, to find the maximum
/// counter ID and maximum expression ID based on their enum variant `id` fields; then, as a
/// safeguard, with `add_missing_operands` set to `true`, to find any other counter or expression
/// IDs referenced by expression operands, if not already seen.
///
/// Ideally, every expression operand in the MIR will have a corresponding Counter or Expression,
/// but since current or future MIR optimizations can theoretically optimize out segments of a
/// MIR, it may not be possible to guarantee this, so the second pass ensures the `CoverageInfo`
/// counts include all referenced IDs.
struct CoverageVisitor {
    info: CoverageInfo,
    add_missing_operands: bool,
}

impl CoverageVisitor {
    // If an expression operand is encountered with an ID outside the range of known counters and
    // expressions, the only way to determine if the ID is a counter ID or an expression ID is to
    // assume a maximum possible counter ID value.
    const MAX_COUNTER_GUARD: u32 = (u32::MAX / 2) + 1;

    #[inline(always)]
    fn update_num_counters(&mut self, counter_id: u32) {
        self.info.num_counters = std::cmp::max(self.info.num_counters, counter_id + 1);
    }

    #[inline(always)]
    fn update_num_expressions(&mut self, expression_id: u32) {
        let expression_index = u32::MAX - expression_id;
        self.info.num_expressions = std::cmp::max(self.info.num_expressions, expression_index + 1);
    }

    fn update_from_expression_operand(&mut self, operand_id: u32) {
        if operand_id >= self.info.num_counters {
            let operand_as_expression_index = u32::MAX - operand_id;
            if operand_as_expression_index >= self.info.num_expressions {
                if operand_id <= Self::MAX_COUNTER_GUARD {
                    self.update_num_counters(operand_id)
                } else {
                    self.update_num_expressions(operand_id)
                }
            }
        }
    }
}

impl Visitor<'_> for CoverageVisitor {
    fn visit_coverage(&mut self, coverage: &Coverage, _location: Location) {
        if self.add_missing_operands {
            match coverage.kind {
                CoverageKind::Expression { lhs, rhs, .. } => {
                    self.update_from_expression_operand(u32::from(lhs));
                    self.update_from_expression_operand(u32::from(rhs));
                }
                _ => {}
            }
        } else {
            match coverage.kind {
                CoverageKind::Counter { id, .. } => {
                    self.update_num_counters(u32::from(id));
                }
                CoverageKind::Expression { id, .. } => {
                    self.update_num_expressions(u32::from(id));
                }
                _ => {}
            }
        }
    }
}

fn coverageinfo_from_mir<'tcx>(tcx: TyCtxt<'tcx>, def_id: DefId) -> CoverageInfo {
    let mir_body = tcx.optimized_mir(def_id);

    let mut coverage_visitor = CoverageVisitor {
        info: CoverageInfo { num_counters: 0, num_expressions: 0 },
        add_missing_operands: false,
    };

    coverage_visitor.visit_body(mir_body);

    coverage_visitor.add_missing_operands = true;
    coverage_visitor.visit_body(mir_body);

    coverage_visitor.info
}
/// Inserts `StatementKind::Coverage` statements that either instrument the binary with injected
/// counters, via intrinsic `llvm.instrprof.increment`, and/or inject metadata used during codegen
/// to construct the coverage map.
pub struct InstrumentCoverage;

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

/// A BasicCoverageBlock (BCB) represents the maximal-length sequence of CFG (MIR) BasicBlocks
/// without conditional branches.
///
/// The BCB allows coverage analysis to be performed on a simplified projection of the underlying
/// MIR CFG, without altering the original CFG. Note that running the MIR `SimplifyCfg` transform,
/// is not sufficient, and therefore not necessary, since the BCB-based CFG projection is a more
/// aggressive simplification. For example:
///
///   * The BCB CFG projection ignores (trims) branches not relevant to coverage, such as unwind-
///     related code that is injected by the Rust compiler but has no physical source code to
///     count. This also means a BasicBlock with a `Call` terminator can be merged into its
///     primary successor target block, in the same BCB.
///   * Some BasicBlock terminators support Rust-specific concerns--like borrow-checking--that are
///     not relevant to coverage analysis. `FalseUnwind`, for example, can be treated the same as
///     a `Goto`, and merged with its successor into the same BCB.
///
/// Each BCB with at least one computed `CoverageSpan` will have no more than one `Counter`.
/// In some cases, a BCB's execution count can be computed by `CounterExpression`. Additional
/// disjoint `CoverageSpan`s in a BCB can also be counted by `CounterExpression` (by adding `ZERO`
/// to the BCB's primary counter or expression).
///
/// Dominator/dominated relationships (which are fundamental to the coverage analysis algorithm)
/// between two BCBs can be computed using the `mir::Body` `dominators()` with any `BasicBlock`
/// member of each BCB. (For consistency, BCB's use the first `BasicBlock`, also referred to as the
/// `bcb_leader_bb`.)
///
/// The BCB CFG projection is critical to simplifying the coverage analysis by ensuring graph
/// path-based queries (`is_dominated_by()`, `predecessors`, `successors`, etc.) have branch
/// (control flow) significance.
#[derive(Debug, Clone)]
struct BasicCoverageBlock {
    pub blocks: Vec<BasicBlock>,
}

impl BasicCoverageBlock {
    pub fn leader_bb(&self) -> BasicBlock {
        self.blocks[0]
    }

    pub fn id(&self) -> String {
        format!(
            "@{}",
            self.blocks
                .iter()
                .map(|bb| bb.index().to_string())
                .collect::<Vec<_>>()
                .join(ID_SEPARATOR)
        )
    }
}

struct BasicCoverageBlocks {
    vec: IndexVec<BasicBlock, Option<BasicCoverageBlock>>,
}

impl BasicCoverageBlocks {
    pub fn from_mir(mir_body: &mir::Body<'tcx>) -> Self {
        let mut basic_coverage_blocks =
            BasicCoverageBlocks { vec: IndexVec::from_elem_n(None, mir_body.basic_blocks().len()) };
        basic_coverage_blocks.extract_from_mir(mir_body);
        basic_coverage_blocks
    }

    pub fn iter(&self) -> impl Iterator<Item = &BasicCoverageBlock> {
        self.vec.iter().filter_map(|bcb| bcb.as_ref())
    }

    pub fn num_nodes(&self) -> usize {
        self.vec.len()
    }

    pub fn extract_from_mir(&mut self, mir_body: &mir::Body<'tcx>) {
        // Traverse the CFG but ignore anything following an `unwind`
        let cfg_without_unwind = ShortCircuitPreorder::new(&mir_body, |term_kind| {
            let mut successors = term_kind.successors();
            match &term_kind {
                // SwitchInt successors are never unwind, and all of them should be traversed.

                // NOTE: TerminatorKind::FalseEdge targets from SwitchInt don't appear to be
                // helpful in identifying unreachable code. I did test the theory, but the following
                // changes were not beneficial. (I assumed that replacing some constants with
                // non-deterministic variables might effect which blocks were targeted by a
                // `FalseEdge` `imaginary_target`. It did not.)
                //
                // Also note that, if there is a way to identify BasicBlocks that are part of the
                // MIR CFG, but not actually reachable, here are some other things to consider:
                //
                // Injecting unreachable code regions will probably require computing the set
                // difference between the basic blocks found without filtering out unreachable
                // blocks, and the basic blocks found with the filter; then computing the
                // `CoverageSpans` without the filter; and then injecting `Counter`s or
                // `CounterExpression`s for blocks that are not unreachable, or injecting
                // `Unreachable` code regions otherwise. This seems straightforward, but not
                // trivial.
                //
                // Alternatively, we might instead want to leave the unreachable blocks in
                // (bypass the filter here), and inject the counters. This will result in counter
                // values of zero (0) for unreachable code (and, notably, the code will be displayed
                // with a red background by `llvm-cov show`).
                //
                // TerminatorKind::SwitchInt { .. } => {
                //     let some_imaginary_target = successors.clone().find_map(|&successor| {
                //         let term = mir_body.basic_blocks()[successor].terminator();
                //         if let TerminatorKind::FalseEdge { imaginary_target, .. } = term.kind {
                //             if mir_body.predecessors()[imaginary_target].len() == 1 {
                //                 return Some(imaginary_target);
                //             }
                //         }
                //         None
                //     });
                //     if let Some(imaginary_target) = some_imaginary_target {
                //         box successors.filter(move |&&successor| successor != imaginary_target)
                //     } else {
                //         box successors
                //     }
                // }
                //
                // Note this also required changing the closure signature for the
                // `ShortCurcuitPreorder` to:
                //
                // F: Fn(&'tcx TerminatorKind<'tcx>) -> Box<dyn Iterator<Item = &BasicBlock> + 'a>,
                TerminatorKind::SwitchInt { .. } => successors,

                // For all other kinds, return only the first successor, if any, and ignore unwinds
                _ => successors.next().into_iter().chain(&[]),
            }
        });

        // Walk the CFG using a Preorder traversal, which starts from `START_BLOCK` and follows
        // each block terminator's `successors()`. Coverage spans must map to actual source code,
        // so compiler generated blocks and paths can be ignored. To that end the CFG traversal
        // intentionally omits unwind paths.
        let mut blocks = Vec::new();
        for (bb, data) in cfg_without_unwind {
            if let Some(last) = blocks.last() {
                let predecessors = &mir_body.predecessors()[bb];
                if predecessors.len() > 1 || !predecessors.contains(last) {
                    // The `bb` has more than one _incoming_ edge, and should start its own
                    // `BasicCoverageBlock`. (Note, the `blocks` vector does not yet include `bb`;
                    // it contains a sequence of one or more sequential blocks with no intermediate
                    // branches in or out. Save these as a new `BasicCoverageBlock` before starting
                    // the new one.)
                    self.add_basic_coverage_block(blocks.split_off(0));
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
            blocks.push(bb);

            let term = data.terminator();

            match term.kind {
                TerminatorKind::Return { .. }
                | TerminatorKind::Abort
                | TerminatorKind::Assert { .. }
                | TerminatorKind::Yield { .. }
                | TerminatorKind::SwitchInt { .. } => {
                    // The `bb` has more than one _outgoing_ edge, or exits the function. Save the
                    // current sequence of `blocks` gathered to this point, as a new
                    // `BasicCoverageBlock`.
                    self.add_basic_coverage_block(blocks.split_off(0));
                    debug!("  because term.kind = {:?}", term.kind);
                    // Note that this condition is based on `TerminatorKind`, even though it
                    // theoretically boils down to `successors().len() != 1`; that is, either zero
                    // (e.g., `Return`, `Abort`) or multiple successors (e.g., `SwitchInt`), but
                    // since the Coverage graph (the BCB CFG projection) ignores things like unwind
                    // branches (which exist in the `Terminator`s `successors()` list) checking the
                    // number of successors won't work.
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

        if !blocks.is_empty() {
            // process any remaining blocks into a final `BasicCoverageBlock`
            self.add_basic_coverage_block(blocks.split_off(0));
            debug!("  because the end of the CFG was reached while traversing");
        }
    }

    fn add_basic_coverage_block(&mut self, blocks: Vec<BasicBlock>) {
        let leader_bb = blocks[0];
        let bcb = BasicCoverageBlock { blocks };
        debug!("adding BCB: {:?}", bcb);
        self.vec[leader_bb] = Some(bcb);
    }
}

impl std::ops::Index<BasicBlock> for BasicCoverageBlocks {
    type Output = BasicCoverageBlock;

    fn index(&self, index: BasicBlock) -> &Self::Output {
        self.vec[index].as_ref().expect("is_some if BasicBlock is a BasicCoverageBlock leader")
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
                    source_range_no_file(tcx, &span),
                    bb.index(),
                    stmt_index,
                    stmt
                )
            }
            Self::Terminator(bb, span) => {
                let term = mir_body[bb].terminator();
                format!(
                    "{}: @{}.{}: {:?}",
                    source_range_no_file(tcx, &span),
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

/// Returns a simple string representation of a `TerminatorKind` variant, indenpendent of any
/// values it might hold.
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
    pub span: Span,
    pub bcb_leader_bb: BasicBlock,
    pub coverage_statements: Vec<CoverageStatement>,
    pub is_closure: bool,
}

impl CoverageSpan {
    pub fn for_statement(
        statement: &Statement<'tcx>,
        span: Span,
        bcb: &BasicCoverageBlock,
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
            bcb_leader_bb: bcb.leader_bb(),
            coverage_statements: vec![CoverageStatement::Statement(bb, span, stmt_index)],
            is_closure,
        }
    }

    pub fn for_terminator(span: Span, bcb: &BasicCoverageBlock, bb: BasicBlock) -> Self {
        Self {
            span,
            bcb_leader_bb: bcb.leader_bb(),
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
    pub fn is_mergeable(&self, other: &Self) -> bool {
        self.is_in_same_bcb(other) && !(self.is_closure || other.is_closure)
    }

    #[inline]
    pub fn is_in_same_bcb(&self, other: &Self) -> bool {
        self.bcb_leader_bb == other.bcb_leader_bb
    }

    pub fn format_coverage_statements(
        &self,
        tcx: TyCtxt<'tcx>,
        mir_body: &'a mir::Body<'tcx>,
    ) -> String {
        let mut sorted_coverage_statements = self.coverage_statements.clone();
        sorted_coverage_statements.sort_unstable_by_key(|covstmt| match *covstmt {
            CoverageStatement::Statement(bb, _, index) => (bb, index),
            CoverageStatement::Terminator(bb, _) => (bb, usize::MAX),
        });
        sorted_coverage_statements
            .iter()
            .map(|covstmt| covstmt.format(tcx, mir_body))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

struct Instrumentor<'a, 'tcx> {
    pass_name: &'a str,
    tcx: TyCtxt<'tcx>,
    mir_body: &'a mut mir::Body<'tcx>,
    body_span: Span,
    basic_coverage_blocks: BasicCoverageBlocks,
    coverage_counters: CoverageCounters,
}

impl<'a, 'tcx> Instrumentor<'a, 'tcx> {
    fn new(pass_name: &'a str, tcx: TyCtxt<'tcx>, mir_body: &'a mut mir::Body<'tcx>) -> Self {
        let hir_body = hir_body(tcx, mir_body.source.def_id());
        let body_span = hir_body.value.span;
        let function_source_hash = hash_mir_source(tcx, hir_body);
        let basic_coverage_blocks = BasicCoverageBlocks::from_mir(mir_body);
        Self {
            pass_name,
            tcx,
            mir_body,
            body_span,
            basic_coverage_blocks,
            coverage_counters: CoverageCounters::new(function_source_hash),
        }
    }

    fn inject_counters(&'a mut self) {
        let tcx = self.tcx;
        let source_map = tcx.sess.source_map();
        let mir_source = self.mir_body.source;
        let def_id = mir_source.def_id();
        let body_span = self.body_span;

        debug!("instrumenting {:?}, span: {}", def_id, source_map.span_to_string(body_span));

        ////////////////////////////////////////////////////
        // Compute `CoverageSpan`s from the `BasicCoverageBlocks`.
        let coverage_spans = CoverageSpans::generate_coverage_spans(
            &self.mir_body,
            body_span,
            &self.basic_coverage_blocks,
        );

        if pretty::dump_enabled(tcx, self.pass_name, def_id) {
            dump_coverage_spanview(
                tcx,
                self.mir_body,
                &self.basic_coverage_blocks,
                self.pass_name,
                &coverage_spans,
            );
        }

        self.inject_coverage_span_counters(coverage_spans);
    }

    /// Inject a counter for each `CoverageSpan`. There can be multiple `CoverageSpan`s for a given
    /// BCB, but only one actual counter needs to be incremented per BCB. `bcb_counters` maps each
    /// `bcb` to its `Counter`, when injected. Subsequent `CoverageSpan`s for a BCB that already has
    /// a `Counter` will inject an `Expression` instead, and compute its value by adding `ZERO` to
    /// the BCB `Counter` value.
    fn inject_coverage_span_counters(&mut self, coverage_spans: Vec<CoverageSpan>) {
        let tcx = self.tcx;
        let source_map = tcx.sess.source_map();
        let body_span = self.body_span;
        let source_file = source_map.lookup_source_file(body_span.lo());
        let file_name = Symbol::intern(&source_file.name.to_string());

        let mut bb_counters = IndexVec::from_elem_n(None, self.mir_body.basic_blocks().len());
        for CoverageSpan { span, bcb_leader_bb: bb, .. } in coverage_spans {
            if let Some(&counter_operand) = bb_counters[bb].as_ref() {
                let expression = self.coverage_counters.make_expression(
                    counter_operand,
                    Op::Add,
                    ExpressionOperandId::ZERO,
                );
                debug!(
                    "Injecting counter expression {:?} at: {:?}:\n{}\n==========",
                    expression,
                    span,
                    source_map.span_to_snippet(span).expect("Error getting source for span"),
                );
                let code_region = make_code_region(file_name, &source_file, span, body_span);
                inject_statement(self.mir_body, expression, bb, Some(code_region));
            } else {
                let counter = self.coverage_counters.make_counter();
                debug!(
                    "Injecting counter {:?} at: {:?}:\n{}\n==========",
                    counter,
                    span,
                    source_map.span_to_snippet(span).expect("Error getting source for span"),
                );
                let counter_operand = counter.as_operand_id();
                bb_counters[bb] = Some(counter_operand);
                let code_region = make_code_region(file_name, &source_file, span, body_span);
                inject_statement(self.mir_body, counter, bb, Some(code_region));
            }
        }
    }
}

/// Generates the MIR pass `CoverageSpan`-specific spanview dump file.
fn dump_coverage_spanview(
    tcx: TyCtxt<'tcx>,
    mir_body: &mir::Body<'tcx>,
    basic_coverage_blocks: &BasicCoverageBlocks,
    pass_name: &str,
    coverage_spans: &Vec<CoverageSpan>,
) {
    let mir_source = mir_body.source;
    let def_id = mir_source.def_id();

    let span_viewables = span_viewables(tcx, mir_body, basic_coverage_blocks, &coverage_spans);
    let mut file = pretty::create_dump_file(tcx, "html", None, pass_name, &0, mir_source)
        .expect("Unexpected error creating MIR spanview HTML file");
    let crate_name = tcx.crate_name(def_id.krate);
    let item_name = tcx.def_path(def_id).to_filename_friendly_no_crate();
    let title = format!("{}.{} - Coverage Spans", crate_name, item_name);
    spanview::write_document(tcx, def_id, span_viewables, &title, &mut file)
        .expect("Unexpected IO error dumping coverage spans as HTML");
}

/// Converts the computed `BasicCoverageBlock`s into `SpanViewable`s.
fn span_viewables(
    tcx: TyCtxt<'tcx>,
    mir_body: &mir::Body<'tcx>,
    basic_coverage_blocks: &BasicCoverageBlocks,
    coverage_spans: &Vec<CoverageSpan>,
) -> Vec<SpanViewable> {
    let mut span_viewables = Vec::new();
    for coverage_span in coverage_spans {
        let tooltip = coverage_span.format_coverage_statements(tcx, mir_body);
        let CoverageSpan { span, bcb_leader_bb: bb, .. } = coverage_span;
        let bcb = &basic_coverage_blocks[*bb];
        let id = bcb.id();
        let leader_bb = bcb.leader_bb();
        span_viewables.push(SpanViewable { bb: leader_bb, span: *span, id, tooltip });
    }
    span_viewables
}

/// Manages the counter and expression indexes/IDs to generate `CoverageKind` components for MIR
/// `Coverage` statements.
struct CoverageCounters {
    function_source_hash: u64,
    next_counter_id: u32,
    num_expressions: u32,
}

impl CoverageCounters {
    pub fn new(function_source_hash: u64) -> Self {
        Self {
            function_source_hash,
            next_counter_id: CounterValueReference::START.as_u32(),
            num_expressions: 0,
        }
    }

    pub fn make_counter(&mut self) -> CoverageKind {
        CoverageKind::Counter {
            function_source_hash: self.function_source_hash,
            id: self.next_counter(),
        }
    }

    pub fn make_expression(
        &mut self,
        lhs: ExpressionOperandId,
        op: Op,
        rhs: ExpressionOperandId,
    ) -> CoverageKind {
        let id = self.next_expression();
        CoverageKind::Expression { id, lhs, op, rhs }
    }

    /// Counter IDs start from one and go up.
    fn next_counter(&mut self) -> CounterValueReference {
        assert!(self.next_counter_id < u32::MAX - self.num_expressions);
        let next = self.next_counter_id;
        self.next_counter_id += 1;
        CounterValueReference::from(next)
    }

    /// Expression IDs start from u32::MAX and go down because a Expression can reference
    /// (add or subtract counts) of both Counter regions and Expression regions. The counter
    /// expression operand IDs must be unique across both types.
    fn next_expression(&mut self) -> InjectedExpressionId {
        assert!(self.next_counter_id < u32::MAX - self.num_expressions);
        let next = u32::MAX - self.num_expressions;
        self.num_expressions += 1;
        InjectedExpressionId::from(next)
    }
}
fn inject_statement(
    mir_body: &mut mir::Body<'tcx>,
    counter_kind: CoverageKind,
    bb: BasicBlock,
    some_code_region: Option<CodeRegion>,
) {
    debug!(
        "  injecting statement {:?} for {:?} at code region: {:?}",
        counter_kind, bb, some_code_region
    );
    let data = &mut mir_body[bb];
    let source_info = data.terminator().source_info;
    let statement = Statement {
        source_info,
        kind: StatementKind::Coverage(box Coverage {
            kind: counter_kind,
            code_region: some_code_region,
        }),
    };
    data.statements.push(statement);
}

/// Converts the initial set of `CoverageSpan`s (one per MIR `Statement` or `Terminator`) into a
/// minimal set of `CoverageSpan`s, using the BCB CFG to determine where it is safe and useful to:
///
///  * Remove duplicate source code coverage regions
///  * Merge spans that represent continuous (both in source code and control flow), non-branching
///    execution
///  * Carve out (leave uncovered) any span that will be counted by another MIR (notably, closures)
pub struct CoverageSpans<'a, 'tcx> {
    /// The MIR, used to look up `BasicBlockData`.
    mir_body: &'a mir::Body<'tcx>,

    /// A snapshot of the MIR CFG dominators before injecting any coverage statements.
    dominators: Dominators<BasicBlock>,

    /// A `Span` covering the function body of the MIR (typically from left curly brace to right
    /// curly brace).
    body_span: Span,

    /// The BasicCoverageBlock Control Flow Graph (BCB CFG).
    basic_coverage_blocks: &'a BasicCoverageBlocks,

    /// The initial set of `CoverageSpan`s, sorted by `Span` (`lo` and `hi`) and by relative
    /// dominance between the `BasicCoverageBlock`s of equal `Span`s.
    sorted_spans_iter: Option<std::vec::IntoIter<CoverageSpan>>,

    /// The current `CoverageSpan` to compare to its `prev`, to possibly merge, discard, force the
    /// discard of the `prev` (and or `pending_dups`), or keep both (with `prev` moved to
    /// `pending_dups`). If `curr` is not discarded or merged, it becomes `prev` for the next
    /// iteration.
    some_curr: Option<CoverageSpan>,

    /// The original `span` for `curr`, in case the `curr` span is modified.
    curr_original_span: Span,

    /// The CoverageSpan from a prior iteration; typically assigned from that iteration's `curr`.
    /// If that `curr` was discarded, `prev` retains its value from the previous iteration.
    some_prev: Option<CoverageSpan>,

    /// Assigned from `curr_original_span` from the previous iteration.
    prev_original_span: Span,

    /// One or more `CoverageSpan`s with the same `Span` but different `BasicCoverageBlock`s, and
    /// no `BasicCoverageBlock` in this list dominates another `BasicCoverageBlock` in the list.
    /// If a new `curr` span also fits this criteria (compared to an existing list of
    /// `pending_dups`), that `curr` `CoverageSpan` moves to `prev` before possibly being added to
    /// the `pending_dups` list, on the next iteration. As a result, if `prev` and `pending_dups`
    /// have the same `Span`, the criteria for `pending_dups` holds for `prev` as well: a `prev`
    /// with a matching `Span` does not dominate any `pending_dup` and no `pending_dup` dominates a
    /// `prev` with a matching `Span`)
    pending_dups: Vec<CoverageSpan>,

    /// The final `CoverageSpan`s to add to the coverage map. A `Counter` or `Expression`
    /// will also be injected into the MIR for each `CoverageSpan`.
    refined_spans: Vec<CoverageSpan>,
}

impl<'a, 'tcx> CoverageSpans<'a, 'tcx> {
    fn generate_coverage_spans(
        mir_body: &'a mir::Body<'tcx>,
        body_span: Span,
        basic_coverage_blocks: &'a BasicCoverageBlocks,
    ) -> Vec<CoverageSpan> {
        let dominators = mir_body.dominators();
        let mut coverage_spans = CoverageSpans {
            mir_body,
            dominators,
            body_span,
            basic_coverage_blocks,
            sorted_spans_iter: None,
            refined_spans: Vec::with_capacity(basic_coverage_blocks.num_nodes() * 2),
            some_curr: None,
            curr_original_span: Span::with_root_ctxt(BytePos(0), BytePos(0)),
            some_prev: None,
            prev_original_span: Span::with_root_ctxt(BytePos(0), BytePos(0)),
            pending_dups: Vec::new(),
        };

        let sorted_spans = coverage_spans.mir_to_initial_sorted_coverage_spans();

        coverage_spans.sorted_spans_iter = Some(sorted_spans.into_iter());
        coverage_spans.some_prev = coverage_spans.sorted_spans_iter.as_mut().unwrap().next();
        coverage_spans.prev_original_span =
            coverage_spans.some_prev.as_ref().expect("at least one span").span;

        coverage_spans.to_refined_spans()
    }

    /// Generate a minimal set of `CoverageSpan`s, each representing a contiguous code region to be
    /// counted.
    ///
    /// The basic steps are:
    ///
    /// 1. Extract an initial set of spans from the `Statement`s and `Terminator`s of each
    ///    `BasicCoverageBlock`.
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
    fn mir_to_initial_sorted_coverage_spans(&self) -> Vec<CoverageSpan> {
        let mut initial_spans = Vec::<CoverageSpan>::with_capacity(self.mir_body.num_nodes() * 2);
        for bcb in self.basic_coverage_blocks.iter() {
            for coverage_span in self.bcb_to_initial_coverage_spans(bcb) {
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
                        self.dominators.rank_partial_cmp(b.bcb_leader_bb, a.bcb_leader_bb)
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

        initial_spans
    }

    /// Iterate through the sorted `CoverageSpan`s, and return the refined list of merged and
    /// de-duplicated `CoverageSpan`s.
    fn to_refined_spans(mut self) -> Vec<CoverageSpan> {
        while self.next_coverage_span() {
            if self.curr().is_mergeable(self.prev()) {
                debug!("  same bcb (and neither is a closure), merge with prev={:?}", self.prev());
                let prev = self.take_prev();
                self.curr_mut().merge_from(prev);
            // Note that curr.span may now differ from curr_original_span
            } else if self.prev_ends_before_curr() {
                debug!(
                    "  different bcbs and disjoint spans, so keep curr for next iter, and add \
                    prev={:?}",
                    self.prev()
                );
                let prev = self.take_prev();
                self.refined_spans.push(prev);
            } else if self.prev().is_closure {
                // drop any equal or overlapping span (`curr`) and keep `prev` to test again in the
                // next iter
                debug!(
                    "  curr overlaps a closure (prev). Drop curr and keep prev for next iter. \
                    prev={:?}",
                    self.prev()
                );
                self.discard_curr();
            } else if self.curr().is_closure {
                self.carve_out_span_for_closure();
            } else if self.prev_original_span == self.curr().span {
                // Note that this compares the new span to `prev_original_span`, which may not
                // be the full `prev.span` (if merged during the previous iteration).
                self.hold_pending_dups_unless_dominated();
            } else {
                self.cutoff_prev_at_overlapping_curr();
            }
        }

        debug!("    AT END, adding last prev={:?}", self.prev());
        let prev = self.take_prev();
        let CoverageSpans {
            mir_body, basic_coverage_blocks, pending_dups, mut refined_spans, ..
        } = self;
        for dup in pending_dups {
            debug!("    ...adding at least one pending dup={:?}", dup);
            refined_spans.push(dup);
        }
        refined_spans.push(prev);

        // Remove `CoverageSpan`s with empty spans ONLY if the empty `CoverageSpan`s BCB also has at
        // least one other non-empty `CoverageSpan`.
        let mut has_coverage = BitSet::new_empty(basic_coverage_blocks.num_nodes());
        for covspan in &refined_spans {
            if !covspan.span.is_empty() {
                has_coverage.insert(covspan.bcb_leader_bb);
            }
        }
        refined_spans.retain(|covspan| {
            !(covspan.span.is_empty()
                && is_goto(&mir_body[covspan.bcb_leader_bb].terminator().kind)
                && has_coverage.contains(covspan.bcb_leader_bb))
        });

        // Remove `CoverageSpan`s derived from closures, originally added to ensure the coverage
        // regions for the current function leave room for the closure's own coverage regions
        // (injected separately, from the closure's own MIR).
        refined_spans.retain(|covspan| !covspan.is_closure);
        refined_spans
    }

    // Generate a set of `CoverageSpan`s from the filtered set of `Statement`s and `Terminator`s of
    // the `BasicBlock`(s) in the given `BasicCoverageBlock`. One `CoverageSpan` is generated
    // for each `Statement` and `Terminator`. (Note that subsequent stages of coverage analysis will
    // merge some `CoverageSpan`s, at which point a `CoverageSpan` may represent multiple
    // `Statement`s and/or `Terminator`s.)
    fn bcb_to_initial_coverage_spans(&self, bcb: &BasicCoverageBlock) -> Vec<CoverageSpan> {
        bcb.blocks
            .iter()
            .map(|bbref| {
                let bb = *bbref;
                let data = &self.mir_body[bb];
                data.statements
                    .iter()
                    .enumerate()
                    .filter_map(move |(index, statement)| {
                        filtered_statement_span(statement, self.body_span).map(|span| {
                            CoverageSpan::for_statement(statement, span, bcb, bb, index)
                        })
                    })
                    .chain(
                        filtered_terminator_span(data.terminator(), self.body_span)
                            .map(|span| CoverageSpan::for_terminator(span, bcb, bb)),
                    )
            })
            .flatten()
            .collect()
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
    ///   * the previous `curr` span (which is now `prev`) was not a duplicate of the pending_dups
    ///     (in which case there should be at least two spans in `pending_dups`); or
    ///   * the `span` of `prev` was modified by `curr_mut().merge_from(prev)` (in which case
    ///     `pending_dups` could have as few as one span)
    /// In either case, no more spans will match the span of `pending_dups`, so
    /// add the `pending_dups` if they don't overlap `curr`, and clear the list.
    fn check_pending_dups(&mut self) {
        if let Some(dup) = self.pending_dups.last() {
            if dup.span != self.prev().span {
                debug!(
                    "    SAME spans, but pending_dups are NOT THE SAME, so BCBs matched on \
                    previous iteration, or prev started a new disjoint span"
                );
                if dup.span.hi() <= self.curr().span.lo() {
                    let pending_dups = self.pending_dups.split_off(0);
                    for dup in pending_dups.into_iter() {
                        debug!("    ...adding at least one pending={:?}", dup);
                        self.refined_spans.push(dup);
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
        while let Some(curr) = self.sorted_spans_iter.as_mut().unwrap().next() {
            debug!("FOR curr={:?}", curr);
            if self.prev_starts_after_next(&curr) {
                debug!(
                    "  prev.span starts after curr.span, so curr will be dropped (skipping past \
                    closure?); prev={:?}",
                    self.prev()
                );
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
            debug!("  prev overlaps a closure. Adding span for pre_closure={:?}", pre_closure);
            if !pending_dups.is_empty() {
                for mut dup in pending_dups.iter().cloned() {
                    dup.span = dup.span.with_hi(left_cutoff);
                    debug!("    ...and at least one pre_closure dup={:?}", dup);
                    self.refined_spans.push(dup);
                }
            }
            self.refined_spans.push(pre_closure);
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

    /// Called if `curr.span` equals `prev_original_span` (and potentially equal to all
    /// `pending_dups` spans, if any); but keep in mind, `prev.span` may start at a `Span.lo()` that
    /// is less than (further left of) `prev_original_span.lo()`.
    ///
    /// When two `CoverageSpan`s have the same `Span`, dominated spans can be discarded; but if
    /// neither `CoverageSpan` dominates the other, both (or possibly more than two) are held,
    /// until their disposition is determined. In this latter case, the `prev` dup is moved into
    /// `pending_dups` so the new `curr` dup can be moved to `prev` for the next iteration.
    fn hold_pending_dups_unless_dominated(&mut self) {
        // Equal coverage spans are ordered by dominators before dominated (if any), so it should be
        // impossible for `curr` to dominate any previous `CoverageSpan`.
        debug_assert!(!self.span_bcb_is_dominated_by(self.prev(), self.curr()));

        let initial_pending_count = self.pending_dups.len();
        if initial_pending_count > 0 {
            let mut pending_dups = self.pending_dups.split_off(0);
            pending_dups.retain(|dup| !self.span_bcb_is_dominated_by(self.curr(), dup));
            self.pending_dups.append(&mut pending_dups);
            if self.pending_dups.len() < initial_pending_count {
                debug!(
                    "  discarded {} of {} pending_dups that dominated curr",
                    initial_pending_count - self.pending_dups.len(),
                    initial_pending_count
                );
            }
        }

        if self.span_bcb_is_dominated_by(self.curr(), self.prev()) {
            debug!(
                "  different bcbs but SAME spans, and prev dominates curr. Discard prev={:?}",
                self.prev()
            );
            self.cutoff_prev_at_overlapping_curr();
        // If one span dominates the other, assocate the span with the code from the dominated
        // block only (`curr`), and discard the overlapping portion of the `prev` span. (Note
        // that if `prev.span` is wider than `prev_original_span`, a `CoverageSpan` will still
        // be created for `prev`s block, for the non-overlapping portion, left of `curr.span`.)
        //
        // For example:
        //     match somenum {
        //         x if x < 1 => { ... }
        //     }...
        //
        // The span for the first `x` is referenced by both the pattern block (every time it is
        // evaluated) and the arm code (only when matched). The counter will be applied only to
        // the dominated block. This allows coverage to track and highlight things like the
        // assignment of `x` above, if the branch is matched, making `x` available to the arm
        // code; and to track and highlight the question mark `?` "try" operator at the end of
        // a function call returning a `Result`, so the `?` is covered when the function returns
        // an `Err`, and not counted as covered if the function always returns `Ok`.
        } else {
            // Save `prev` in `pending_dups`. (`curr` will become `prev` in the next iteration.)
            // If the `curr` CoverageSpan is later discarded, `pending_dups` can be discarded as
            // well; but if `curr` is added to refined_spans, the `pending_dups` will also be added.
            debug!(
                "  different bcbs but SAME spans, and neither dominates, so keep curr for \
                next iter, and, pending upcoming spans (unless overlapping) add prev={:?}",
                self.prev()
            );
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
        debug!(
            "  different bcbs, overlapping spans, so ignore/drop pending and only add prev \
            if it has statements that end before curr; prev={:?}",
            self.prev()
        );
        if self.pending_dups.is_empty() {
            let curr_span = self.curr().span;
            self.prev_mut().cutoff_statements_at(curr_span.lo());
            if self.prev().coverage_statements.is_empty() {
                debug!("  ... no non-overlapping statements to add");
            } else {
                debug!("  ... adding modified prev={:?}", self.prev());
                let prev = self.take_prev();
                self.refined_spans.push(prev);
            }
        } else {
            // with `pending_dups`, `prev` cannot have any statements that don't overlap
            self.pending_dups.clear();
        }
    }

    fn span_bcb_is_dominated_by(&self, covspan: &CoverageSpan, dom_covspan: &CoverageSpan) -> bool {
        self.dominators.is_dominated_by(covspan.bcb_leader_bb, dom_covspan.bcb_leader_bb)
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
            Some(function_source_span(statement.source_info.span, body_span))
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
        // For `FalseEdge`, only the `real` branch is taken, so it is similar to a `Goto`.
        // FIXME(richkadel): Note that `Goto` was moved to it's own match arm, for the reasons
        // described below. Add tests to confirm whether or not similar cases also apply to
        // `FalseEdge`.
        | TerminatorKind::FalseEdge { .. } => None,

        // FIXME(richkadel): Note that `Goto` was initially filtered out (by returning `None`, as
        // with the `TerminatorKind`s above) because its `Span` was way to broad to be beneficial,
        // and, at the time, `Goto` didn't seem to provide any additional contributions to the
        // coverage analysis. Upon further review, `Goto` terminated blocks do appear to benefit
        // the coverage analysis, and the BCB CFG. To overcome the issues with the `Spans`, the
        // coverage algorithms--and the final coverage map generation--include some exceptional
        // behaviors.
        //
        // `Goto`s are often the targets of `SwitchInt` branches, and certain important
        // optimizations to replace some `Counter`s with `Expression`s require a separate
        // `BasicCoverageBlock` for each branch, to support the `Counter`, when needed.
        //
        // Also, some test cases showed that `Goto` terminators, and to some degree their `Span`s,
        // provided useful context for coverage, such as to count and show when `if` blocks
        // _without_ `else` blocks execute the `false` case (counting when the body of the `if`
        // was _not_ taken). In these cases, the `Goto` span is ultimately given a `CoverageSpan`
        // of 1 character, at the end of it's original `Span`.
        //
        // However, in other cases, a visible `CoverageSpan` is not wanted, but the `Goto`
        // block must still be counted (for example, to contribute its count to an `Expression`
        // that reports the execution count for some other block). In these cases, the code region
        // is set to `None`.
        TerminatorKind::Goto { .. } => {
            Some(function_source_span(terminator.source_info.span.shrink_to_hi(), body_span))
        }

        // Retain spans from all other terminators
        TerminatorKind::Resume
        | TerminatorKind::Abort
        | TerminatorKind::Return
        | TerminatorKind::Call { .. }
        | TerminatorKind::Yield { .. }
        | TerminatorKind::GeneratorDrop
        | TerminatorKind::FalseUnwind { .. }
        | TerminatorKind::InlineAsm { .. } => {
            Some(function_source_span(terminator.source_info.span, body_span))
        }
    }
}

#[inline(always)]
fn function_source_span(span: Span, body_span: Span) -> Span {
    let span = original_sp(span, body_span).with_ctxt(SyntaxContext::root());
    if body_span.contains(span) { span } else { body_span }
}

#[inline(always)]
fn is_goto(term_kind: &TerminatorKind<'tcx>) -> bool {
    match term_kind {
        TerminatorKind::Goto { .. } => true,
        _ => false,
    }
}

/// Convert the Span into its file name, start line and column, and end line and column
fn make_code_region(
    file_name: Symbol,
    source_file: &Lrc<SourceFile>,
    span: Span,
    body_span: Span,
) -> CodeRegion {
    let (start_line, mut start_col) = source_file.lookup_file_pos(span.lo());
    let (end_line, end_col) = if span.hi() == span.lo() {
        let (end_line, mut end_col) = (start_line, start_col);
        // Extend an empty span by one character so the region will be counted.
        let CharPos(char_pos) = start_col;
        if span.hi() == body_span.hi() {
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
