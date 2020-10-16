use crate::transform::MirPass;
use crate::util::pretty;
use crate::util::spanview::{self, SpanViewable};

use rustc_data_structures::fingerprint::Fingerprint;
use rustc_data_structures::graph::dominators::Dominators;
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

use std::cmp::Ordering;

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
        // If the InstrumentCoverage pass is called on promoted MIRs, skip them.
        // See: https://github.com/rust-lang/rust/pull/73011#discussion_r438317601
        if mir_body.source.promoted.is_some() {
            trace!(
                "InstrumentCoverage skipped for {:?} (already promoted for Miri evaluation)",
                mir_body.source.def_id()
            );
            return;
        }

        let hir_id = tcx.hir().local_def_id_to_hir_id(mir_body.source.def_id().expect_local());
        let is_fn_like = FnLikeNode::from_node(tcx.hir().get(hir_id)).is_some();

        // Only instrument functions, methods, and closures (not constants since they are evaluated
        // at compile time by Miri).
        // FIXME(#73156): Handle source code coverage in const eval
        if !is_fn_like {
            trace!(
                "InstrumentCoverage skipped for {:?} (not an FnLikeNode)",
                mir_body.source.def_id(),
            );
            return;
        }
        // FIXME(richkadel): By comparison, the MIR pass `ConstProp` includes associated constants,
        // with functions, methods, and closures. I assume Miri is used for associated constants as
        // well. If not, we may need to include them here too.

        trace!("InstrumentCoverage starting for {:?}", mir_body.source.def_id());
        Instrumentor::new(&self.name(), tcx, mir_body).inject_counters();
        trace!("InstrumentCoverage starting for {:?}", mir_body.source.def_id());
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
        self.vec.iter().filter_map(|option| option.as_ref())
    }

    fn extract_from_mir(&mut self, mir_body: &mir::Body<'tcx>) {
        // Traverse the CFG but ignore anything following an `unwind`
        let cfg_without_unwind = ShortCircuitPreorder::new(mir_body, |term_kind| {
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
                let stmt = &mir_body.basic_blocks()[bb].statements[stmt_index];
                format!(
                    "{}: @{}[{}]: {:?}",
                    spanview::source_range_no_file(tcx, &span),
                    bb.index(),
                    stmt_index,
                    stmt
                )
            }
            Self::Terminator(bb, span) => {
                let term = mir_body.basic_blocks()[bb].terminator();
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
    bcb_leader_bb: BasicBlock,
    coverage_statements: Vec<CoverageStatement>,
    is_closure: bool,
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

    pub fn for_terminator(span: Span, bcb: &'a BasicCoverageBlock, bb: BasicBlock) -> Self {
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

    pub fn is_dominated_by(
        &self,
        other: &CoverageSpan,
        dominators: &Dominators<BasicBlock>,
    ) -> bool {
        debug_assert!(!self.is_in_same_bcb(other));
        dominators.is_dominated_by(self.bcb_leader_bb, other.bcb_leader_bb)
    }

    pub fn is_mergeable(&self, other: &Self) -> bool {
        self.is_in_same_bcb(other) && !(self.is_closure || other.is_closure)
    }

    pub fn is_in_same_bcb(&self, other: &Self) -> bool {
        self.bcb_leader_bb == other.bcb_leader_bb
    }
}

struct Instrumentor<'a, 'tcx> {
    pass_name: &'a str,
    tcx: TyCtxt<'tcx>,
    mir_body: &'a mut mir::Body<'tcx>,
    hir_body: &'tcx rustc_hir::Body<'tcx>,
    dominators: Option<Dominators<BasicBlock>>,
    basic_coverage_blocks: Option<BasicCoverageBlocks>,
    function_source_hash: Option<u64>,
    next_counter_id: u32,
    num_expressions: u32,
}

impl<'a, 'tcx> Instrumentor<'a, 'tcx> {
    fn new(pass_name: &'a str, tcx: TyCtxt<'tcx>, mir_body: &'a mut mir::Body<'tcx>) -> Self {
        let hir_body = hir_body(tcx, mir_body.source.def_id());
        Self {
            pass_name,
            tcx,
            mir_body,
            hir_body,
            dominators: None,
            basic_coverage_blocks: None,
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

    fn dominators(&self) -> &Dominators<BasicBlock> {
        self.dominators.as_ref().expect("dominators must be initialized before calling")
    }

    fn basic_coverage_blocks(&self) -> &BasicCoverageBlocks {
        self.basic_coverage_blocks
            .as_ref()
            .expect("basic_coverage_blocks must be initialized before calling")
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

    fn inject_counters(&mut self) {
        let tcx = self.tcx;
        let source_map = tcx.sess.source_map();
        let def_id = self.mir_body.source.def_id();
        let mir_body = &self.mir_body;
        let body_span = self.body_span();
        let source_file = source_map.lookup_source_file(body_span.lo());
        let file_name = Symbol::intern(&source_file.name.to_string());

        debug!("instrumenting {:?}, span: {}", def_id, source_map.span_to_string(body_span));

        self.dominators.replace(mir_body.dominators());
        self.basic_coverage_blocks.replace(BasicCoverageBlocks::from_mir(mir_body));

        let coverage_spans = self.coverage_spans();

        let span_viewables = if pretty::dump_enabled(tcx, self.pass_name, def_id) {
            Some(self.span_viewables(&coverage_spans))
        } else {
            None
        };

        // Inject a counter for each `CoverageSpan`. There can be multiple `CoverageSpan`s for a
        // given BCB, but only one actual counter needs to be incremented per BCB. `bb_counters`
        // maps each `bcb_leader_bb` to its `Counter`, when injected. Subsequent `CoverageSpan`s
        // for a BCB that already has a `Counter` will inject a `CounterExpression` instead, and
        // compute its value by adding `ZERO` to the BCB `Counter` value.
        let mut bb_counters = IndexVec::from_elem_n(None, mir_body.basic_blocks().len());
        for CoverageSpan { span, bcb_leader_bb: bb, .. } in coverage_spans {
            if let Some(&counter_operand) = bb_counters[bb].as_ref() {
                let expression =
                    self.make_expression(counter_operand, Op::Add, ExpressionOperandId::ZERO);
                debug!(
                    "Injecting counter expression {:?} at: {:?}:\n{}\n==========",
                    expression,
                    span,
                    source_map.span_to_snippet(span).expect("Error getting source for span"),
                );
                self.inject_statement(file_name, &source_file, expression, span, bb);
            } else {
                let counter = self.make_counter();
                debug!(
                    "Injecting counter {:?} at: {:?}:\n{}\n==========",
                    counter,
                    span,
                    source_map.span_to_snippet(span).expect("Error getting source for span"),
                );
                let counter_operand = counter.as_operand_id();
                bb_counters[bb] = Some(counter_operand);
                self.inject_statement(file_name, &source_file, counter, span, bb);
            }
        }

        if let Some(span_viewables) = span_viewables {
            let mut file = pretty::create_dump_file(
                tcx,
                "html",
                None,
                self.pass_name,
                &0,
                self.mir_body.source,
            )
            .expect("Unexpected error creating MIR spanview HTML file");
            let crate_name = tcx.crate_name(def_id.krate);
            let item_name = tcx.def_path(def_id).to_filename_friendly_no_crate();
            let title = format!("{}.{} - Coverage Spans", crate_name, item_name);
            spanview::write_document(tcx, def_id, span_viewables, &title, &mut file)
                .expect("Unexpected IO error dumping coverage spans as HTML");
        }
    }

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
        coverage_kind: CoverageKind,
        span: Span,
        block: BasicBlock,
    ) {
        let code_region = make_code_region(file_name, source_file, span);
        debug!("  injecting statement {:?} covering {:?}", coverage_kind, code_region);

        let data = &mut self.mir_body[block];
        let source_info = data.terminator().source_info;
        let statement = Statement {
            source_info,
            kind: StatementKind::Coverage(box Coverage { kind: coverage_kind, code_region }),
        };
        data.statements.push(statement);
    }

    /// Converts the computed `BasicCoverageBlock`s into `SpanViewable`s.
    fn span_viewables(&self, coverage_spans: &Vec<CoverageSpan>) -> Vec<SpanViewable> {
        let tcx = self.tcx;
        let mir_body = &self.mir_body;
        let mut span_viewables = Vec::new();
        for coverage_span in coverage_spans {
            let bcb = self.bcb_from_coverage_span(coverage_span);
            let CoverageSpan { span, bcb_leader_bb: bb, coverage_statements, .. } = coverage_span;
            let id = bcb.id();
            let mut sorted_coverage_statements = coverage_statements.clone();
            sorted_coverage_statements.sort_unstable_by_key(|covstmt| match *covstmt {
                CoverageStatement::Statement(bb, _, index) => (bb, index),
                CoverageStatement::Terminator(bb, _) => (bb, usize::MAX),
            });
            let tooltip = sorted_coverage_statements
                .iter()
                .map(|covstmt| covstmt.format(tcx, mir_body))
                .collect::<Vec<_>>()
                .join("\n");
            span_viewables.push(SpanViewable { bb: *bb, span: *span, id, tooltip });
        }
        span_viewables
    }

    #[inline(always)]
    fn bcb_from_coverage_span(&self, coverage_span: &CoverageSpan) -> &BasicCoverageBlock {
        &self.basic_coverage_blocks()[coverage_span.bcb_leader_bb]
    }

    #[inline(always)]
    fn body_span(&self) -> Span {
        self.hir_body.value.span
    }

    // Generate a set of `CoverageSpan`s from the filtered set of `Statement`s and `Terminator`s of
    // the `BasicBlock`(s) in the given `BasicCoverageBlock`. One `CoverageSpan` is generated for
    // each `Statement` and `Terminator`. (Note that subsequent stages of coverage analysis will
    // merge some `CoverageSpan`s, at which point a `CoverageSpan` may represent multiple
    // `Statement`s and/or `Terminator`s.)
    fn extract_spans(&self, bcb: &'a BasicCoverageBlock) -> Vec<CoverageSpan> {
        let body_span = self.body_span();
        let mir_basic_blocks = self.mir_body.basic_blocks();
        bcb.blocks
            .iter()
            .map(|bbref| {
                let bb = *bbref;
                let data = &mir_basic_blocks[bb];
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
    fn coverage_spans(&self) -> Vec<CoverageSpan> {
        let mut initial_spans =
            Vec::<CoverageSpan>::with_capacity(self.mir_body.basic_blocks().len() * 2);
        for bcb in self.basic_coverage_blocks().iter() {
            for coverage_span in self.extract_spans(bcb) {
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
                        self.dominators().rank_partial_cmp(b.bcb_leader_bb, a.bcb_leader_bb)
                    }
                } else {
                    // Sort hi() in reverse order so shorter spans are attempted after longer spans.
                    // This guarantees that, if a `prev` span overlaps, and is not equal to, a `curr`
                    // span, the prev span either extends further left of the curr span, or they
                    // start at the same position and the prev span extends further right of the end
                    // of the curr span.
                    b.span.hi().partial_cmp(&a.span.hi())
                }
            } else {
                a.span.lo().partial_cmp(&b.span.lo())
            }
            .unwrap()
        });

        let refinery = CoverageSpanRefinery::from_sorted_spans(initial_spans, self.dominators());
        refinery.to_refined_spans()
    }
}

struct CoverageSpanRefinery<'a> {
    sorted_spans_iter: std::vec::IntoIter<CoverageSpan>,
    dominators: &'a Dominators<BasicBlock>,
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
        dominators: &'a Dominators<BasicBlock>,
    ) -> Self {
        let refined_spans = Vec::with_capacity(sorted_spans.len());
        let mut sorted_spans_iter = sorted_spans.into_iter();
        let prev = sorted_spans_iter.next().expect("at least one span");
        let prev_original_span = prev.span;
        Self {
            sorted_spans_iter,
            dominators,
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
                self.add_refined_span(prev);
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
                self.hold_pending_dups_unless_dominated();
            } else {
                self.cutoff_prev_at_overlapping_curr();
            }
        }
        debug!("    AT END, adding last prev={:?}", self.prev());
        let pending_dups = self.pending_dups.split_off(0);
        for dup in pending_dups.into_iter() {
            debug!("    ...adding at least one pending dup={:?}", dup);
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
                debug!(
                    "    SAME spans, but pending_dups are NOT THE SAME, so BCBs matched on \
                    previous iteration, or prev started a new disjoint span"
                );
                if dup.span.hi() <= self.curr().span.lo() {
                    let pending_dups = self.pending_dups.split_off(0);
                    for dup in pending_dups.into_iter() {
                        debug!("    ...adding at least one pending={:?}", dup);
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
        debug_assert!(!self.prev().is_dominated_by(self.curr(), self.dominators));

        if self.curr().is_dominated_by(&self.prev(), self.dominators) {
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
            debug!(
                "  different bcbs but SAME spans, and prev dominates curr. Drop curr and \
                keep prev for next iter. prev={:?}",
                self.prev()
            );
            self.discard_curr();
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
            if it has statements that end before curr={:?}",
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
        TerminatorKind::Unreachable // Unreachable blocks are not connected to the CFG
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
