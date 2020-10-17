use crate::transform::MirPass;
use crate::util::generic_graphviz::GraphvizWriter;
use crate::util::pretty;
use crate::util::spanview::{self, SpanViewable};

use rustc_data_structures::fingerprint::Fingerprint;
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::graph::dominators::{self, Dominators};
use rustc_data_structures::graph::{self, GraphSuccessors, WithNumNodes, WithStartNode};
use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_data_structures::sync::Lrc;
use rustc_index::bit_set::BitSet;
use rustc_index::vec::{Idx, IndexVec};
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
use std::ops::{Index, IndexMut};
use std::lazy::SyncOnceCell;

const ID_SEPARATOR: &str = ",";

const RUSTC_COVERAGE_DEBUG_OPTIONS: &str = "RUSTC_COVERAGE_DEBUG_OPTIONS";

/// Inserts `StatementKind::Coverage` statements that either instrument the binary with injected
/// counters, via intrinsic `llvm.instrprof.increment`, and/or inject metadata used during codegen
/// to construct the coverage map.
pub struct InstrumentCoverage;

#[derive(Debug, Clone)]
struct DebugOptions {
    allow_unused_expressions: bool,
    simplify_expressions: bool,
    counter_format: ExpressionFormat,
}

impl DebugOptions {
    fn new() -> Self {
        let mut allow_unused_expressions = true;
        let mut simplify_expressions = false;
        let mut counter_format = ExpressionFormat::default();

        if let Ok(env_debug_options) = std::env::var(RUSTC_COVERAGE_DEBUG_OPTIONS) {
            for setting_str in env_debug_options.replace(" ", "").replace("-", "_").split(",") {
                let mut setting = setting_str.splitn(2, "=");
                match setting.next() {
                    Some(option) if option == "allow_unused_expressions" => {
                        allow_unused_expressions = bool_option_val(option, setting.next());
                        debug!("{} env option `allow_unused_expressions` is set to {}", RUSTC_COVERAGE_DEBUG_OPTIONS, allow_unused_expressions);
                    }
                    Some(option) if option == "simplify_expressions" => {
                        simplify_expressions = bool_option_val(option, setting.next());
                        debug!("{} env option `simplify_expressions` is set to {}", RUSTC_COVERAGE_DEBUG_OPTIONS, simplify_expressions);
                    }
                    Some(option) if option == "counter_format" => {
                        if let Some(strval) = setting.next() {
                            counter_format = counter_format_option_val(strval);
                            debug!("{} env option `counter_format` is set to {:?}", RUSTC_COVERAGE_DEBUG_OPTIONS, counter_format);
                        } else {
                            bug!("`{}` option in environment variable {} requires one or more plus-separated choices (a non-empty subset of `id+block+operation`)", option, RUSTC_COVERAGE_DEBUG_OPTIONS);
                        }
                    }
                    Some("") => {},
                    Some(invalid) => bug!("Unsupported setting `{}` in environment variable {}", invalid, RUSTC_COVERAGE_DEBUG_OPTIONS),
                    None => {},
                }
            }
        }

        Self {
            allow_unused_expressions,
            simplify_expressions,
            counter_format,
        }
    }
}

fn debug_options<'a>() -> &'a DebugOptions {
    static DEBUG_OPTIONS: SyncOnceCell<DebugOptions> = SyncOnceCell::new();

    &DEBUG_OPTIONS.get_or_init(|| DebugOptions::new())
}

fn bool_option_val(option: &str, some_strval: Option<&str>) -> bool {
    if let Some(val) = some_strval {
        if vec!["yes", "y", "on", "true"].contains(&val) {
            true
        } else if vec!["no", "n", "off", "false"].contains(&val) {
            false
        } else {
            bug!("Unsupported value `{}` for option `{}` in environment variable {}", option, val, RUSTC_COVERAGE_DEBUG_OPTIONS)
        }
    } else {
        true
    }
}

fn counter_format_option_val(strval: &str) -> ExpressionFormat {
    let mut counter_format = ExpressionFormat {
        id: false,
        block: false,
        operation: false,
    };
    let components = strval.splitn(3, "+");
    for component in components {
        match component {
            "id" => counter_format.id = true,
            "block" => counter_format.block = true,
            "operation" => counter_format.operation = true,
            _ => bug!("Unsupported counter_format choice `{}` in environment variable {}", component, RUSTC_COVERAGE_DEBUG_OPTIONS),
        }
    }
    counter_format
}

#[derive(Debug, Clone)]
struct ExpressionFormat {
    id: bool,
    block: bool,
    operation: bool,
}

impl Default for ExpressionFormat {
    fn default() -> Self {
        Self {
            id: false,
            block: true,
            operation: false,
        }
    }
}

#[derive(Debug)]
struct DebugCounter {
    counter_kind: CoverageKind,
    some_block_label: Option<String>,
}

impl DebugCounter {
    fn new(counter_kind: CoverageKind, some_block_label: Option<String>) -> Self {
        Self {
            counter_kind,
            some_block_label,
        }
    }
}

struct DebugCounters {
    some_counters: Option<FxHashMap<ExpressionOperandId, DebugCounter>>,
}

impl DebugCounters {
    pub fn new() -> Self {
        Self {
            some_counters: None,
        }
    }

    pub fn enable(&mut self) {
        self.some_counters.replace(FxHashMap::default());
    }

    pub fn is_enabled(&mut self) -> bool {
        self.some_counters.is_some()
    }

    pub fn add_counter(&mut self, counter_kind: &CoverageKind, some_block_label: Option<String>) {
        if let Some(counters) = &mut self.some_counters {
            let id: ExpressionOperandId = match *counter_kind {
                CoverageKind::Counter { id, .. } => id.into(),
                | CoverageKind::Expression { id, .. } => id.into(),
                _ => bug!("the given `CoverageKind` is not an counter or expression: {:?}", counter_kind),
            };
            counters.insert(id.into(), DebugCounter::new(counter_kind.clone(), some_block_label)).expect_none("attempt to add the same counter_kind to DebugCounters more than once");
        }
    }

    pub fn some_block_label(&self, operand: ExpressionOperandId) -> Option<&String> {
        self.some_counters.as_ref().map_or(None, |counters| counters.get(&operand).map_or(None, |debug_counter| debug_counter.some_block_label.as_ref()))
    }

    pub fn format_counter(&self, counter_kind: &CoverageKind) -> String {
        match *counter_kind {
            CoverageKind::Counter { .. } => format!("Counter({})", self.format_counter_kind(counter_kind)),
            CoverageKind::Expression { .. } => format!("Expression({})", self.format_counter_kind(counter_kind)),
            CoverageKind::Unreachable { .. } => "Unreachable".to_owned(),
        }
    }

    fn format_counter_kind(&self, counter_kind: &CoverageKind) -> String {
        let counter_format = &debug_options().counter_format;
        if let CoverageKind::Expression { id, lhs, op, rhs } = *counter_kind {
            if counter_format.operation {
                return format!(
                    "{}{} {} {}",
                    if counter_format.id || self.some_counters.is_none() {
                        format!("#{} = ", id.index() )
                    } else {
                        String::new()
                    },
                    self.format_operand(lhs),
                    if op == Op::Add { "+" } else { "-" },
                    self.format_operand(rhs),
                );
            }
        }

        let id: ExpressionOperandId = match *counter_kind {
            CoverageKind::Counter { id, .. } => id.into(),
            | CoverageKind::Expression { id, .. } => id.into(),
            _ => bug!("the given `CoverageKind` is not an counter or expression: {:?}", counter_kind),
        };
        if self.some_counters.is_some() && (counter_format.block || !counter_format.id) {
            let counters = self.some_counters.as_ref().unwrap();
            if let Some(DebugCounter { some_block_label: Some(block_label), .. }) = counters.get(&id.into()) {
                return if counter_format.id {
                    format!("{}#{}", block_label, id.index())
                } else {
                    format!("{}", block_label)
                }
            }
        }
        format!("#{}", id.index())
    }

    fn format_operand(&self, operand: ExpressionOperandId) -> String {
        if operand.index() == 0 {
            return String::from("0");
        }
        if let Some(counters) = &self.some_counters {
            if let Some(DebugCounter { counter_kind, some_block_label }) = counters.get(&operand) {
                if let CoverageKind::Expression { .. } = counter_kind {
                    if let Some(block_label) = some_block_label {
                        if debug_options().counter_format.block {
                            return format!("{}:({})", block_label, self.format_counter_kind(counter_kind));
                        }
                    }
                    return format!("({})", self.format_counter_kind(counter_kind));
                }
                return format!("{}", self.format_counter_kind(counter_kind));
            }
        }
        format!("#{}", operand.index().to_string())
    }
}

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
/// In some cases, a BCB's execution count can be computed by `Expression`. Additional
/// disjoint `CoverageSpan`s in a BCB can also be counted by `Expression` (by adding `ZERO`
/// to the BCB's primary counter or expression).
///
/// The BCB CFG is critical to simplifying the coverage analysis by ensuring graph path-based
/// queries (`is_dominated_by()`, `predecessors`, `successors`, etc.) have branch (control flow)
/// significance.
#[derive(Debug, Clone)]
struct BasicCoverageBlockData {
    basic_blocks: Vec<BasicBlock>,
    counter_kind: Option<CoverageKind>,
    edge_counter_from_bcbs: Option<FxHashMap<BasicCoverageBlock, CoverageKind>>,
}

impl BasicCoverageBlockData {
    pub fn from(basic_blocks: Vec<BasicBlock>) -> Self {
        assert!(basic_blocks.len() > 0);
        Self { basic_blocks, counter_kind: None, edge_counter_from_bcbs: None }
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
    pub fn terminator<'a, 'tcx>(&self, mir_body: &'a mir::Body<'tcx>) -> &'a Terminator<'tcx> {
        &mir_body[self.last_bb()].terminator()
    }

    #[inline(always)]
    pub fn set_counter(&mut self, counter_kind: CoverageKind) -> ExpressionOperandId {
        debug_assert!(self.edge_counter_from_bcbs.is_none() || counter_kind.is_expression(), "attempt to add a `Counter` to a BCB target with existing incoming edge counters");
        let operand = counter_kind.as_operand_id();
        self.counter_kind
            .replace(counter_kind)
            .expect_none("attempt to set a BasicCoverageBlock coverage counter more than once");
        operand
    }

    #[inline(always)]
    pub fn counter(&self) -> Option<&CoverageKind> {
        self.counter_kind.as_ref()
    }

    #[inline(always)]
    pub fn take_counter(&mut self) -> Option<CoverageKind> {
        self.counter_kind.take()
    }

    #[inline(always)]
    pub fn set_edge_counter_from(&mut self, from_bcb: BasicCoverageBlock, counter_kind: CoverageKind) -> ExpressionOperandId {
        debug_assert!(self.counter_kind.as_ref().map_or(true, |c| c.is_expression()), "attempt to add an incoming edge counter from {:?} when the target BCB already has a `Counter`", from_bcb);
        let operand = counter_kind.as_operand_id();
        self.edge_counter_from_bcbs
        .get_or_insert_with(|| FxHashMap::default())
            .insert(from_bcb, counter_kind)
            .expect_none("attempt to set an edge counter more than once");
        operand
    }

    #[inline(always)]
    pub fn edge_counter_from(&self, from_bcb: BasicCoverageBlock) -> Option<&CoverageKind> {
        if let Some(edge_counter_from_bcbs) = &self.edge_counter_from_bcbs {
            edge_counter_from_bcbs.get(&from_bcb)
        } else {
            None
        }
    }

    #[inline(always)]
    pub fn take_edge_counters(&mut self) -> Option<impl Iterator<Item = (BasicCoverageBlock, CoverageKind)>> {
        self.edge_counter_from_bcbs.take().map_or(None, |m|Some(m.into_iter()))
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

    pub fn to_string_sections(
        &self,
        tcx: TyCtxt<'tcx>,
        mir_body: &mir::Body<'tcx>,
        debug_counters: &DebugCounters,
        some_coverage_spans_with_counters: Option<&Vec<(CoverageSpan, CoverageKind)>>,
        some_dependency_counters: Option<&Vec<CoverageKind>>,
        some_intermediate_expressions: Option<&Vec<CoverageKind>>,
    ) -> Vec<String> {
        let len = self.basic_blocks.len();
        let mut sections = Vec::new();
        if let Some(collect_intermediate_expressions) = some_intermediate_expressions {
            sections.push(
                collect_intermediate_expressions
                    .iter()
                    .map(|expression| format!("Intermediate {}", debug_counters.format_counter(expression)))
                    .collect::<Vec<_>>()
                    .join("\n"),
            );
        }
        if let Some(coverage_spans_with_counters) = some_coverage_spans_with_counters {
            sections.push(
                coverage_spans_with_counters
                    .iter()
                    .map(|(covspan, counter)| format!("{} at {}", debug_counters.format_counter(counter), covspan.format(tcx, mir_body)))
                    .collect::<Vec<_>>()
                    .join("\n"),
            );
        }
        if let Some(dependency_counters) = some_dependency_counters {
            sections.push(
                format!(
                    "Non-coverage counters:\n  {}",
                    dependency_counters
                        .iter()
                        .map(|counter| debug_counters.format_counter(counter))
                        .collect::<Vec<_>>()
                        .join("  \n"),
                )
            );
        }
        if let Some(counter_kind) = &self.counter_kind {
            sections.push(format!("{:?}", counter_kind));
        }
        let non_term_blocks = self.basic_blocks[0..len - 1]
            .iter()
            .map(|&bb| format!("{:?}: {}", bb, term_type(&mir_body[bb].terminator().kind)))
            .collect::<Vec<_>>();
        if non_term_blocks.len() > 0 {
            sections.push(non_term_blocks.join("\n"));
        }
        sections.push(format!(
            "{:?}: {}",
            self.basic_blocks.last().unwrap(),
            term_type(&self.terminator(mir_body).kind)
        ));
        sections
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
                let bcb_successors = bcb_filtered_successors(&mir_body, &bcb_data.terminator(mir_body).kind)
// TODO(richkadel):
// MAKE SURE WE ONLY RETURN THE SAME SUCCESSORS USED WHEN CREATING THE BCB (THE FIRST SUCCESSOR ONLY,
// UNLESS ITS A SWITCHINT).)

// THEN BUILD PREDECESSORS FROM THE SUCCESSORS
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

        Self { bcbs, bb_to_bcb, successors, predecessors }
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
    pub fn iter_enumerated_mut(
        &mut self,
    ) -> impl Iterator<Item = (BasicCoverageBlock, &mut BasicCoverageBlockData)> {
        self.bcbs.iter_enumerated_mut()
    }

    #[inline(always)]
    pub fn bcb_from_bb(&self, bb: BasicBlock) -> Option<BasicCoverageBlock> {
        if bb.index() < self.bb_to_bcb.len() {
            self.bb_to_bcb[bb]
        } else {
            None
        }
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
        self.bcb_from_bb(mir::START_BLOCK).expect("mir::START_BLOCK should be in a BasicCoverageBlock")
    }
}

type BcbSuccessors<'graph> = std::slice::Iter<'graph, BasicCoverageBlock>;

impl graph::WithSuccessors for BasicCoverageBlocks {
    #[inline]
    fn successors(&self, node: Self::Node) -> <Self as GraphSuccessors<'_>>::Iter {
        self.successors[node].iter().cloned()
    }
}

impl<'graph> graph::GraphSuccessors<'graph> for BasicCoverageBlocks {
    type Item = BasicCoverageBlock;
    type Iter = std::iter::Cloned<BcbSuccessors<'graph>>;
}

type BcbPredecessors = Vec<BasicCoverageBlock>;

impl graph::GraphPredecessors<'graph> for BasicCoverageBlocks {
    type Item = BasicCoverageBlock;
    type Iter = std::vec::IntoIter<BasicCoverageBlock>;
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

    pub fn format(&self, tcx: TyCtxt<'tcx>, mir_body: &'a mir::Body<'tcx>) -> String {
        format!(
            "{}\n    {}",
            spanview::source_range_no_file(tcx, &self.span),
            self.format_coverage_statements(tcx, mir_body).replace("\n", "\n    "),
        )
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

/// Maintains separate worklists for each loop in the BasicCoverageBlock CFG, plus one for the
/// BasicCoverageBlocks outside all loops. This supports traversing the BCB CFG in a way that
/// ensures a loop is completely traversed before processing Blocks after the end of the loop.
#[derive(Debug)]
struct TraversalContext {
    /// From one or more backedges returning to a loop header.
    loop_backedges: Option<(Vec<BasicCoverageBlock>, BasicCoverageBlock)>,

    /// worklist, to be traversed, of BasicCoverageBlocks in the loop with the given loop
    /// backedges, such that the loop is the inner inner-most loop containing these
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
    debug_expressions_cache: Option<FxHashMap<ExpressionOperandId, CoverageKind>>,
    debug_counters: DebugCounters,
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
            debug_expressions_cache: None,
            debug_counters: DebugCounters::new(),
        }
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

        let dump_spanview = pretty::dump_enabled(tcx, self.pass_name, def_id);
        let dump_graphviz = tcx.sess.opts.debugging_opts.dump_mir_graphviz;

        if dump_graphviz {
            self.debug_counters.enable();
        }

        let coverage_spans = self.coverage_spans();
        let span_viewables =
            if dump_spanview { Some(self.span_viewables(&coverage_spans)) } else { None };
        let mut collect_intermediate_expressions = Vec::with_capacity(self.basic_coverage_blocks.num_nodes());

        // When debug logging, or generating the coverage graphviz output, initialize the following
        // data structures:
        let mut debug_used_expression_operands = None;
        let mut debug_unused_expressions = None;
        if level_enabled!(tracing::Level::DEBUG) || dump_graphviz {
            debug_used_expression_operands = Some(FxHashMap::default());
            debug_unused_expressions = Some(Vec::new());
            if debug_options().simplify_expressions {
                self.debug_expressions_cache.replace(FxHashMap::default());
            }
            // CAUTION! The `simplify_expressions` option is only helpful for some debugging
            // situations and it can change the generated MIR `Coverage` statements (resulting in
            // differences in behavior when enabled, under `DEBUG`, compared to normal operation and
            // testing).
            //
            // For debugging purposes, it is sometimes helpful to simplify some expression equations:
            //
            //   * `x + (y - x)` becomes just `y`
            //   * `x + (y + 0)` becomes just x + y.
            //
            // Expression dependencies can deeply nested expressions, which can look quite long in
            // printed debug messages and in graphs produced by `-Zdump-graphviz`. In reality, each
            // referenced/nested expression is only present because that value is necessary to
            // compute a counter value for another part of the coverage report. Simplifying expressions
            // Does not result in less `Coverage` statements, so there is very little, if any, benefit
            // to binary size or runtime to simplifying expressions, and adds additional compile-time
            // complexity. Only enable this temporarily, if helpful to parse the debug output.
        }

        // When debugging with BCB graphviz output, initialize additional data structures.
        let mut debug_bcb_to_coverage_spans_with_counters = None;
        let mut debug_bcb_to_dependency_counter = None;
        let mut debug_edge_to_counter = None;
        if dump_graphviz {
            debug_bcb_to_coverage_spans_with_counters = Some(FxHashMap::default());
            debug_bcb_to_dependency_counter = Some(FxHashMap::default());
            debug_edge_to_counter = Some(FxHashMap::default());
        }

        let mut bcbs_with_coverage = BitSet::new_empty(self.basic_coverage_blocks.num_nodes());
        for covspan in &coverage_spans {
            bcbs_with_coverage.insert(covspan.bcb);
        }

        // Analyze the coverage graph (aka, BCB control flow graph), and inject expression-optimized
        // counters.
        self.make_bcb_counters(bcbs_with_coverage, &mut collect_intermediate_expressions);

        // If debugging, add any intermediate expressions (which are not associated with any BCB) to
        // the `debug_used_expression_operands` map.

        if let Some(used_expression_operands) = debug_used_expression_operands.as_mut() {
            for intermediate_expression in &collect_intermediate_expressions {
                if let CoverageKind::Expression { id, lhs, rhs, .. } = *intermediate_expression {
                    used_expression_operands.entry(lhs).or_insert_with(|| Vec::new()).push(id);
                    used_expression_operands.entry(rhs).or_insert_with(|| Vec::new()).push(id);
                }
            }
        }

        // Inject a counter for each `CoverageSpan`. There can be multiple `CoverageSpan`s for a
        // given BCB, but only one actual counter needs to be incremented per BCB. `bb_counters`
        // maps each `bcb` to its `Counter`, when injected. Subsequent `CoverageSpan`s
        // for a BCB that already has a `Counter` will inject an `Expression` instead, and
        // compute its value by adding `ZERO` to the BCB `Counter` value.
        //
        // If debugging, add every BCB `Expression` associated with a `CoverageSpan`s to the
        // `debug_used_expression_operands` map.
        let mut bcb_counters = IndexVec::from_elem_n(None, self.basic_coverage_blocks.num_nodes());
        for covspan in coverage_spans {
            let bcb = covspan.bcb;
            let span = covspan.span;
            let counter_kind = if let Some(&counter_operand) = bcb_counters[bcb].as_ref() {
                self.make_identity_counter(counter_operand)
            } else if let Some(counter_kind) = self.bcb_data_mut(bcb).take_counter() {
                bcb_counters[bcb] = Some(counter_kind.as_operand_id());
                if let Some(used_expression_operands) = debug_used_expression_operands.as_mut() {
                    if let CoverageKind::Expression { id, lhs, rhs, .. } = counter_kind {
                        used_expression_operands.entry(lhs).or_insert_with(|| Vec::new()).push(id);
                        used_expression_operands.entry(rhs).or_insert_with(|| Vec::new()).push(id);
                    }
                }
                counter_kind
            } else {
                bug!("Every BasicCoverageBlock should have a Counter or Expression");
            };
            // TODO(richkadel): uncomment debug!
            // debug!(
            //     "Injecting {} at: {:?}:\n{}\n==========",
            //     self.format_counter(counter_kind),
            //     span,
            //     source_map.span_to_snippet(span).expect("Error getting source for span"),
            // );
            if let Some(bcb_to_coverage_spans_with_counters) = debug_bcb_to_coverage_spans_with_counters.as_mut() {
                bcb_to_coverage_spans_with_counters.entry(bcb).or_insert_with(|| Vec::new()).push((covspan.clone(), counter_kind.clone()));
            }
            let mut code_region = None;
            if span.hi() == body_span.hi() {
                // TODO(richkadel): add a comment if this works
                if let TerminatorKind::Goto { .. } = self.bcb_terminator(bcb).kind {
                    code_region = Some(make_non_reportable_code_region(file_name, &source_file, span));
                }
            }
            if code_region.is_none() {
                code_region = Some(make_code_region(file_name, &source_file, span, body_span));
            };
            self.inject_statement(counter_kind, self.bcb_last_bb(bcb), code_region.unwrap());
        }

        // The previous step looped through the `CoverageSpan`s and injected the counter from the
        // `CoverageSpan`s `BasicCoverageBlock`, removing it from the BCB in the process (via
        // `take_counter()`).
        //
        // Any other counter associated with a `BasicCoverageBlock`, or its incoming edge, but not
        // associated with a `CoverageSpan`, should only exist if the counter is a
        // `Expression` dependency (one of the expression operands). Collect them, and inject
        // the additional counters into the MIR, without a reportable coverage span.
        let mut bcb_counters_without_direct_coverage_spans = Vec::new();
        for (target_bcb, target_bcb_data) in self.basic_coverage_blocks.iter_enumerated_mut() {
            if let Some(counter_kind) = target_bcb_data.take_counter() {
                bcb_counters_without_direct_coverage_spans.push((None, target_bcb, counter_kind));
            }
            if let Some(edge_counters) = target_bcb_data.take_edge_counters() {
                for (from_bcb, counter_kind) in edge_counters {
                    bcb_counters_without_direct_coverage_spans.push((Some(from_bcb), target_bcb, counter_kind));
                }
            }
        }

        // Validate that every BCB or edge counter not directly associated with a coverage span is
        // at least indirectly associated (it is a dependency of a BCB counter that _is_ associated
        // with a coverage span).
        if let Some(used_expression_operands) = debug_used_expression_operands.as_mut() {
            let mut not_validated = bcb_counters_without_direct_coverage_spans.iter().map(|(_, _, counter_kind)| counter_kind).collect::<Vec<_>>();
            let mut validating_count = 0;
            while not_validated.len() != validating_count  {
                let to_validate = not_validated.split_off(0);
                validating_count = to_validate.len();
                for counter_kind in to_validate {
                    if used_expression_operands.contains_key(&counter_kind.as_operand_id()) {
                        if let CoverageKind::Expression { id, lhs, rhs, .. } = *counter_kind {
                            used_expression_operands.entry(lhs).or_insert_with(|| Vec::new()).push(id);
                            used_expression_operands.entry(rhs).or_insert_with(|| Vec::new()).push(id);
                        }
                    } else {
                        not_validated.push(counter_kind);
                    }
                }
            }
        }

//        let (last_line, last_col) = source_file.lookup_file_pos(body_span.hi());
        for (edge_counter_from_bcb, target_bcb, counter_kind) in bcb_counters_without_direct_coverage_spans {
            if let (
                Some(used_expression_operands),
                Some(unused_expressions),
            ) = (
                debug_used_expression_operands.as_ref(),
                debug_unused_expressions.as_mut(),
            ) {
                if !used_expression_operands.contains_key(&counter_kind.as_operand_id()) {
                    unused_expressions.push((counter_kind.clone(), edge_counter_from_bcb, target_bcb));
                }
            }

            match counter_kind {
                CoverageKind::Counter { .. } => {
                    let inject_to_bb = if let Some(from_bcb) = edge_counter_from_bcb {
                        // The MIR edge starts `from_bb` (the outgoing / last BasicBlock in `from_bcb`) and
                        // ends at `to_bb` (the incoming / first BasicBlock in the `target_bcb`; also called
                        // the `leader_bb`).
                        let from_bb = self.bcb_last_bb(from_bcb);
                        let to_bb = self.bcb_leader_bb(target_bcb);

                        debug!(
                            "Edge {:?} (last {:?}) -> {:?} (leader {:?}) requires a new MIR BasicBlock, for unclaimed edge counter {}",
                            edge_counter_from_bcb, from_bb, target_bcb, to_bb, self.format_counter(&counter_kind),
                        );
                        debug!(
                            "  from_bb {:?} has successors: {:?}",
                            from_bb, self.mir_body[from_bb].terminator().successors(),
                        );
                        let span = self.mir_body[from_bb].terminator().source_info.span.shrink_to_hi();
                        let new_bb = self.mir_body.basic_blocks_mut().push(BasicBlockData {
                            statements: vec![], // counter will be injected here
                            terminator: Some(Terminator {
                                source_info: SourceInfo::outermost(span),
                                kind: TerminatorKind::Goto { target: to_bb },
                            }),
                            is_cleanup: false,
                        });
                        let edge_ref = self.mir_body[from_bb].terminator_mut().successors_mut().find(|successor| **successor == to_bb).expect("from_bb should have a successor for to_bb");
                        *edge_ref = new_bb;

                        if let Some(edge_to_counter) = debug_edge_to_counter.as_mut() {
                            debug!("from_bcb={:?} to new_bb={:?} has edge_counter={}",
                                    from_bcb, new_bb, self.format_counter(&counter_kind),
                            );
                            edge_to_counter.insert((from_bcb, new_bb), counter_kind.clone()).expect_none("invalid attempt to insert more than one edge counter for the same edge");
                        }
                        new_bb
                    } else {
                        if let Some(bcb_to_dependency_counter) = debug_bcb_to_dependency_counter.as_mut() {
                            bcb_to_dependency_counter.entry(target_bcb).or_insert_with(|| Vec::new()).push(counter_kind.clone());
                        }
                        let target_bb = self.bcb_last_bb(target_bcb);
                        debug!(
                            "{:?} ({:?}) gets a new Coverage statement for unclaimed counter {}",
                            target_bcb,
                            target_bb,
                            self.format_counter(&counter_kind),
                        );
                        target_bb
                    };
//                    debug!("make_non_reportable_code_region for {:?} at last_line={:?}, last_col={:?}, counter={}",
//                                    inject_to_bb, last_line, last_col, self.format_counter(&counter_kind));
//                    self.inject_statement(counter_kind, inject_to_bb, make_non_reportable_code_region(file_name, last_line, last_col));
                    let span = self.mir_body[inject_to_bb].terminator().source_info.span;
                    debug!("make_non_reportable_code_region for {:?} at span={:?}, counter={}",
                                    inject_to_bb, span, self.format_counter(&counter_kind));
                    self.inject_statement(counter_kind, inject_to_bb, make_non_reportable_code_region(file_name, &source_file, span));
                }
                CoverageKind::Expression { .. } => self.inject_intermediate_expression(counter_kind),
                _ => bug!("CoverageKind should be a counter"),
            }
        }

        if dump_graphviz {
            let bcb_to_coverage_spans_with_counters = debug_bcb_to_coverage_spans_with_counters.expect("graphviz data should exist if dump_graphviz is true");
            let bcb_to_dependency_counter = debug_bcb_to_dependency_counter.expect("graphviz data should exist if dump_graphviz is true");
            let edge_to_counter = debug_edge_to_counter.expect("graphviz data should exist if dump_graphviz is true");
            let graphviz_name = format!("Cov_{}_{}", def_id.krate.index(), def_id.index.index());
            let node_content = |bcb| {
                self.bcb_data(bcb).to_string_sections(
                    tcx,
                    self.mir_body,
                    &self.debug_counters,
                    bcb_to_coverage_spans_with_counters.get(&bcb),
                    bcb_to_dependency_counter.get(&bcb),
                    // collect_intermediate_expressions are injected into the mir::START_BLOCK, so include
                    // them in the first BCB.
                    if bcb.index() == 0 { Some(&collect_intermediate_expressions) } else { None }
                )
            };
            let edge_labels = |from_bcb| {
                let from_terminator = self.bcb_terminator(from_bcb);
                let mut edge_labels = from_terminator.kind.fmt_successor_labels();
                edge_labels.retain(|label| label.to_string() != "unreachable");
                let edge_counters = from_terminator.successors().map(|&successor| {
                    edge_to_counter.get(&(from_bcb, successor))
                });
                edge_labels.iter().zip(edge_counters).map(|(label, some_counter)| {
                    if let Some(counter) = some_counter {
                        format!("{}\n{}", label, self.format_counter(counter))
                    } else {
                        label.to_string()
                    }
                }).collect::<Vec<_>>()
            };
            let mut graphviz_writer = GraphvizWriter::new(
                &self.basic_coverage_blocks,
                &graphviz_name,
                node_content,
                edge_labels,
            );
            if let Some(unused_expressions) = debug_unused_expressions.as_ref() {
                if unused_expressions.len() > 0 {
                    graphviz_writer.set_graph_label(&format!(
                        "Unused expressions:\n  {}",
                        unused_expressions.as_slice().iter().map(|(counter_kind, edge_counter_from_bcb, target_bcb)| {
                            if let Some(from_bcb) = edge_counter_from_bcb.as_ref() {
                                format!(
                                    "{:?}->{:?}: {}",
                                    from_bcb,
                                    target_bcb,
                                    self.format_counter(&counter_kind),
                                )
                            } else {
                                format!(
                                    "{:?}: {}",
                                    target_bcb,
                                    self.format_counter(&counter_kind),
                                )
                            }
                        }).collect::<Vec<_>>().join("\n  ")
                    ));
                }
            }
            let mut file =
                pretty::create_dump_file(tcx, "dot", None, self.pass_name, &0, mir_source)
                    .expect("Unexpected error creating BasicCoverageBlock graphviz DOT file");
            graphviz_writer
                .write_graphviz(tcx, &mut file)
                .expect("Unexpected error writing BasicCoverageBlock graphviz DOT file");
        }

        if let Some(unused_expressions) = debug_unused_expressions.as_ref() {
            for (counter_kind, edge_counter_from_bcb, target_bcb) in unused_expressions {
                let unused_counter_message = if let Some(from_bcb) = edge_counter_from_bcb.as_ref() {
                    format!(
                        "non-coverage edge counter found without a dependent expression, in {:?}->{:?}; counter={}",
                        from_bcb,
                        target_bcb,
                        self.format_counter(&counter_kind),
                    )
                } else {
                    format!(
                        "non-coverage counter found without a dependent expression, in {:?}; counter={}",
                        target_bcb,
                        self.format_counter(&counter_kind),
                    )
                };

                // FIXME(richkadel): Determine if unused expressions can and should be prevented, and
                // if so, it would be a `bug!` if encountered in the future. At the present time,
                // however we do sometimes encounter unused expressions (in only a subset of test cases)
                // even when `simplify_expressions` is disabled. It's not yet clear what causes this, or
                // if it should be allowed in the long term, but as long as the coverage capability
                // still works, generate warning messages only, for now.
                if self.debug_expressions_cache.is_some() || debug_options().allow_unused_expressions {
                    // Note, the debugging option `simplify_expressions`, which initializes the
                    // `debug_expressions_cache` can cause some counters to become unused, and
                    // is not a bug.
                    //
                    // For example, converting `x + (y - x)` to just `y` removes a dependency
                    // on `y - x`. If that expression is not a dependency elsewhere, and if it is
                    // not associated with a `CoverageSpan`, it is now considered `unused`.
                    debug!("WARNING: {}", unused_counter_message);
                } else {
                    bug!("{}", unused_counter_message);
                }
            }
        }

        for intermediate_expression in collect_intermediate_expressions {
            self.inject_intermediate_expression(intermediate_expression);
        }

        // TODO(richkadel):
        //
        // 1. "try" coverage isn't quite right. See `bcb14` in try_error_result.rs.
        //    The `?` character should have its own coverage span (for bcb14, which is otherwise
        //    uncovered) indicating when the error result occurred, and the `main()` function
        //    returned `Err`. It looks like it's because the SwitchInt block includes an Assign
        //    with the exact same span as the `?` span, and since it dominates the `Err` handling
        //    code for `?`, it got the span. This seems wrong here, but the `dominator gets equal spans`
        //    logic was there for the match guard handling.
        //
        //    Maybe I really should reverse that match guard handling. I know it looks weird, but actually
        //    it may be right. The assignment of the value to the guard variable is (maybe?) to make
        //    it available to the arm code?  Well, actually, not (only) in that case since I use it for
        //    the comparison. But then, why doesn't the comparison get it? (Or does it? Maybe it would
        //    get it and that's why it would be counted twice!)
        //
        //    I need to try it again.
        // 2. SwitchInt branches should exclude branches to unreachable blocks.
        // 3. BCB's without CoverageSpans may (??!!??) not need counters unless they are part
        //    of another expression. With the current algorithm, I think the only way they can
        //    be part of an expression is if they are a SwitchInt or a target of a SwitchInt,
        //    AND I think the SwitchInt counter logic may ensure those counters are added,
        //    so we can probably just not add the counters during traversal if there are no
        //    coverage spans.
        // 4. If a BCB gets a counter (because it's a SwitchInt or target of SwitchInt), and
        //    there is no CoverageSpan for that BCB, we still must `inject_statement` the
        //    counter, ... but with what coverage span??? Maybe an empty span at the terminator's
        //    span.lo()?

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

    /// Traverse the BCB CFG and add either a `Counter` or `Expression` to ever BCB, to be
    /// injected with `CoverageSpan`s. `Expressions` have no runtime overhead, so if a viable
    /// expression (adding or subtracting two other counters or expressions) can compute the same
    /// result as an embedded counter, an `Expression` should be used.
    ///
    /// If two `BasicCoverageBlocks` branch from another `BasicCoverageBlock`, one of the branches
    /// can be counted by `Expression` by subtracting the other branch from the branching
    /// block. Otherwise, the `BasicCoverageBlock` executed the least should have the `Counter`.
    /// One way to predict which branch executes the least is by considering loops. A loop is exited
    /// at a branch, so the branch that jumps to a `BasicCoverageBlock` outside the loop is almost
    /// always executed less than the branch that does not exit the loop.
    ///
    /// Returns non-code-span expressions created to represent intermediate values (if required),
    /// such as to add two counters so the result can be subtracted from another counter.
    fn make_bcb_counters(
        &mut self,
        bcbs_with_coverage: BitSet<BasicCoverageBlock>,
        collect_intermediate_expressions: &mut Vec<CoverageKind>,
    ) {
        debug!("make_bcb_counters(): adding a counter or expression to each BasicCoverageBlock");
        let num_bcbs = self.basic_coverage_blocks.num_nodes();
        let mut backedges = IndexVec::from_elem_n(
            Vec::<BasicCoverageBlock>::new(),
            num_bcbs,
        );

        // Identify loops by their backedges
        for (bcb, _) in self.basic_coverage_blocks.iter_enumerated() {
            for &successor in &self.basic_coverage_blocks.successors[bcb] {
                if self.bcb_is_dominated_by(bcb, successor) {
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

        let start_bcb = self.basic_coverage_blocks.start_node();

        // `context_stack` starts with a `TraversalContext` for the main function context (beginning
        // with the `start` BasicCoverageBlock of the function). New worklists are pushed to the top
        // of the stack as loops are entered, and popped off of the stack when a loop's worklist is
        // exhausted.
        let mut context_stack = Vec::new();
        context_stack.push(TraversalContext { loop_backedges: None, worklist: vec![start_bcb] });
        let mut visited = BitSet::new_empty(num_bcbs);

        while let Some(bcb) = {
            // Strip contexts with empty worklists from the top of the stack
            while context_stack.last().map_or(false, |context| context.worklist.is_empty()) {
                context_stack.pop();
            }
            // Pop the next bcb off of the current context_stack. If none, all BCBs were visited.
            context_stack.last_mut().map_or(None, |context| context.worklist.pop())
        } {
            if !visited.insert(bcb) {
                debug!("Already visited: {:?}", bcb);
                continue;
            }
            debug!("Visiting {:?}", bcb);
            if backedges[bcb].len() > 0 {
                debug!("{:?} is a loop header! Start a new TraversalContext...", bcb);
                context_stack.push(TraversalContext {
                    loop_backedges: Some((backedges[bcb].clone(), bcb)),
                    worklist: Vec::new(),
                });
            }

            debug!(
                "{:?} has {} successors:",
                bcb,
                self.basic_coverage_blocks.successors[bcb].len()
            );
            for &successor in &self.basic_coverage_blocks.successors[bcb] {
                for context in context_stack.iter_mut().rev() {
                    if let Some((_, loop_header)) = context.loop_backedges {
                        if self.bcb_is_dominated_by(successor, loop_header) {
                            if self.bcb_successors(successor).len() > 1 {
                                debug!(
                                    "Adding branching successor {:?} to the beginning of the worklist of loop headed by {:?}",
                                    successor, loop_header
                                );
                                context.worklist.insert(0, successor);
                            } else {
                                debug!(
                                    "Adding non-branching successor {:?} to the end of the worklist of loop headed by {:?}",
                                    successor, loop_header
                                );
                                context.worklist.push(successor);
                            }
                            break;
                        }
                    } else {
                        if self.bcb_successors(successor).len() > 1 {
                            debug!("Adding branching successor {:?} to the beginning of the non-loop worklist", successor);
                            context.worklist.insert(0, successor);
                        } else {
                            debug!("Adding non-branching successor {:?} to the end of the non-loop worklist", successor);
                            context.worklist.push(successor);
                        }
                    }
                }
            }

            if !bcbs_with_coverage.contains(bcb) {
                continue;
            }

            let bcb_counter_operand = self.get_or_make_counter_operand(bcb, collect_intermediate_expressions);

            let needs_branch_counters = {
                let successors = self.bcb_successors(bcb);
                successors.len() > 1 && successors.iter().any(|&successor| self.bcb_data(successor).counter().is_none())
            };
            if needs_branch_counters {
                let branching_bcb = bcb;
                let branching_counter_operand = bcb_counter_operand;
                let branches = self.bcb_successors(branching_bcb).clone();

                debug!(
                    "{:?} is branching, with branches: {:?}",
                    branching_bcb,
                    branches
                );

                // At most one of the branches (or its edge, from the branching_bcb,
                // if the branch has multiple incoming edges) can have a counter computed by
                // expression.
                //
                // If at least one of the branches leads outside of a loop (`found_loop_exit` is
                // true), and at least one other branch does not exit the loop (the first of which
                // is captured in `some_reloop_branch`), it's likely any reloop branch will be
                // executed far more often than loop exit branch, making the reloop branch a better
                // candidate for an expression.
                let mut some_reloop_branch = None;
                for context in context_stack.iter().rev() {
                    if let Some((backedge_from_bcbs, _)) = &context.loop_backedges {
                        let mut found_loop_exit = false;
                        for &branch in branches.iter() {
                            if backedge_from_bcbs.iter().any(|&backedge_from_bcb| {
                                self.bcb_is_dominated_by(backedge_from_bcb, branch)
                            }) {
                                // The path from branch leads back to the top of the loop
                                some_reloop_branch = Some(branch);
                            } else {
                                // The path from branch leads outside this loop
                                found_loop_exit = true;
                            }
                            if some_reloop_branch.is_some() && found_loop_exit {
                                break;
                            }
                        }
                        debug!(
                            "found_loop_exit={}, some_reloop_branch={:?}",
                            found_loop_exit, some_reloop_branch
                        );
                        if !found_loop_exit {
                            // No branches exit a loop, so there is no specific recommended branch for
                            // an `Expression`.
                            break;
                        }
                        if some_reloop_branch.is_some() {
                            // A recommended branch for an `Expression` was found.
                            break;
                        }
                        // else all branches exited this loop context, so run the same checks with
                        // the outer loop(s)
                    }
                }

                // Select a branch for the expression, either the recommended `reloop_branch`, or
                // if none was found, select any branch.
                let expression_branch = if let Some(reloop_branch) = some_reloop_branch {
                    debug!("Adding expression to reloop_branch={:?}", reloop_branch);
                    reloop_branch
                } else {
                    let &branch_without_counter = branches
                        .iter()
                        .find(|&&branch| {
                            self.bcb_data(branch).counter().is_none()
                        })
                        .expect("needs_branch_counters was `true` so there should be at least one branch");
                    debug!(
                        "No preferred expression branch. Selected the first branch without a counter. That branch={:?}",
                        branch_without_counter
                    );
                    branch_without_counter
                };

                // Assign a Counter or Expression to each branch, plus additional
                // `Expression`s, as needed, to sum up intermediate results.
                let mut some_sumup_counter_operand = None;
                for branch in branches {
                    if branch != expression_branch {
                        let branch_counter_operand =
                            if self.bcb_has_multiple_incoming_edges(branch) {
                                debug!("{:?} has multiple incoming edges, so adding an edge counter from {:?}", branch, branching_bcb);
                                self.get_or_make_edge_counter_operand(branching_bcb, branch, collect_intermediate_expressions)
                            } else {
                                debug!("{:?} has only one incoming edge (from {:?}), so adding a counter", branch, branching_bcb);
// TODO(richkadel): IS THIS FOR LOOP DUPLICATING WHAT'S IN get_or_make_counter_operand?
                                self.get_or_make_counter_operand(branch, collect_intermediate_expressions)
                            };
                        if let Some(sumup_counter_operand) =
                            some_sumup_counter_operand.replace(branch_counter_operand)
                        {
                            let intermediate_expression = self.make_expression(
                                branch_counter_operand,
                                Op::Add,
                                sumup_counter_operand,
                                || None,
                            );
                            debug!("  new intermediate expression: {}", self.format_counter(&intermediate_expression));
                            let intermediate_expression_operand = intermediate_expression.as_operand_id();
                            collect_intermediate_expressions.push(intermediate_expression);
                            some_sumup_counter_operand.replace(intermediate_expression_operand);
                        }
                    }
                }
                let sumup_counter_operand = some_sumup_counter_operand.expect("sumup_counter_operand should have a value");
                let multiple_incoming_edges = self.bcb_has_multiple_incoming_edges(expression_branch);
                debug!("expression_branch is {:?}, multiple_incoming_edges={}, expression_branch predecessors: {:?}",
                    expression_branch, multiple_incoming_edges, self.bcb_predecessors(expression_branch));
                let expression = self.make_expression(
                    branching_counter_operand,
                    Op::Subtract,
                    sumup_counter_operand,
                    || Some(
                        if multiple_incoming_edges {
                            format!("{:?}->{:?}", branching_bcb, expression_branch)
                        } else {
                            format!("{:?}", expression_branch)
                        }
                    )
                );
                if multiple_incoming_edges {
                    debug!("Edge {:?}->{:?} gets an expression: {}", branching_bcb, expression_branch, self.format_counter(&expression));
                    self.bcb_data_mut(expression_branch).set_edge_counter_from(branching_bcb, expression);
                } else {
                    debug!("{:?} gets an expression: {}", expression_branch, self.format_counter(&expression));
                    self.bcb_data_mut(expression_branch).set_counter(expression);
                }

// TODO(richkadel):
// Who would use this edge counter?
// Since this isn't the bcb counter, what will the bcb counter be?
// (I think the bcb counter will be the sum of all edge counters)
//
// Does this mean that we should assert that a bcb ONLY has either a single bcb counter, or 2 or more edge counters, but not both?
// WELL... not exactly.
// 1. A BCB with edge counters always has an Expression
// 2. A BCB with multiple predecessors always gets an Expression.
//
// ???
// 3. If NO predecessor comes from a branching BCB, then the BCB Expression can be the sum of all predecessor BCB counters
// ???
//
// 4. If (any?) predecessor comes from a branching BCB, assume the incoming edges will
//    ?? all ??
//    need edge counters, and the BCB Expression is the sum of all incoming edge counters.
//
//       THIS (#4) DOESN'T SOUND RIGHT
//
// 5. An incoming edge from a source BCB (from_bcb) with only one successor (so, from_bcb is not a branching BCB) does not
// need a separate outgoing edge counter. In this case, when summing up incoming edge counts for the successor BCB, for
// each edge, use the incoming edge counter if present, or use the counter from the edge SOURCE (and assert the source has
// only one successor)



            }
        }

        debug_assert_eq!(visited.count(), visited.domain_size());
    }

    #[inline]
    fn format_counter(&self, counter_kind: &CoverageKind) -> String {
        self.debug_counters.format_counter(counter_kind)
    }

    #[inline]
    fn bcb_leader_bb(&self, bcb: BasicCoverageBlock) -> BasicBlock {
        self.bcb_data(bcb).leader_bb()
    }

    #[inline]
    fn bcb_last_bb(&self, bcb: BasicCoverageBlock) -> BasicBlock {
        self.bcb_data(bcb).last_bb()
    }

    #[inline]
    fn bcb_terminator(&self, bcb: BasicCoverageBlock) -> &Terminator<'tcx> {
        self.bcb_data(bcb).terminator(self.mir_body)
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
    fn bcb_predecessors(&self, bcb: BasicCoverageBlock) -> &BcbPredecessors {
        &self.basic_coverage_blocks.predecessors[bcb]
    }

    #[inline]
    fn bcb_successors(&self, bcb: BasicCoverageBlock) -> &Vec<BasicCoverageBlock> {
        &self.basic_coverage_blocks.successors[bcb]
    }

    #[inline]
    fn bcb_has_multiple_incoming_edges(&self, bcb: BasicCoverageBlock) -> bool {
        self.bcb_predecessors(bcb).len() > 1
    }

    #[inline]
    fn bcb_is_dominated_by(&self, node: BasicCoverageBlock, dom: BasicCoverageBlock) -> bool {
        self.bcb_dominators.is_dominated_by(node, dom)
    }

    fn get_or_make_counter_operand(&mut self, bcb: BasicCoverageBlock, collect_intermediate_expressions: &mut Vec<CoverageKind>) -> ExpressionOperandId {
        if let Some(counter_kind) = self.basic_coverage_blocks[bcb].counter() {
            debug!("  {:?} already has a counter: {}", bcb, self.format_counter(counter_kind));
            counter_kind.as_operand_id()
        } else {
            if self.bcb_has_multiple_incoming_edges(bcb) {
                let mut predecessors = self.bcb_predecessors(bcb).clone().into_iter();
                let first_edge_counter_operand = self.get_or_make_edge_counter_operand(predecessors.next().unwrap(), bcb, collect_intermediate_expressions);
                let mut some_sumup_edge_counter_operand = None;
                for predecessor in predecessors {
                    let edge_counter_operand = self.get_or_make_edge_counter_operand(predecessor, bcb, collect_intermediate_expressions);
                    if let Some(sumup_edge_counter_operand) =
                        some_sumup_edge_counter_operand.replace(edge_counter_operand)
                    {
                        let intermediate_expression = self.make_expression(
                            sumup_edge_counter_operand,
                            Op::Add,
                            edge_counter_operand,
                            || None,
                        );
                        debug!("  new intermediate expression: {}", self.format_counter(&intermediate_expression));
                        let intermediate_expression_operand = intermediate_expression.as_operand_id();
                        collect_intermediate_expressions.push(intermediate_expression);
                        some_sumup_edge_counter_operand.replace(intermediate_expression_operand);
                    }
                }
                let counter_kind = self.make_expression(
                    first_edge_counter_operand,
                    Op::Add,
                    some_sumup_edge_counter_operand.unwrap(),
                    || Some(format!("{:?}", bcb))
                );
                debug!("  {:?} gets a new counter (sum of predecessor counters): {}", bcb, self.format_counter(&counter_kind));
                self.basic_coverage_blocks[bcb].set_counter(counter_kind)
            } else {
                let counter_kind = self.make_counter(|| Some(format!("{:?}", bcb)));
                debug!("  {:?} gets a new counter: {}", bcb, self.format_counter(&counter_kind));
                self.basic_coverage_blocks[bcb].set_counter(counter_kind)
            }
        }
    }

    fn get_or_make_edge_counter_operand(&mut self, from_bcb: BasicCoverageBlock, to_bcb: BasicCoverageBlock, collect_intermediate_expressions: &mut Vec<CoverageKind>) -> ExpressionOperandId {
        let successors = self.bcb_successors(from_bcb).iter();
        if successors.len() > 1 {
            if let Some(counter_kind) = self.basic_coverage_blocks[to_bcb].edge_counter_from(from_bcb) {
                debug!("  Edge {:?}->{:?} already has a counter: {}", from_bcb, to_bcb, self.format_counter(counter_kind));
                counter_kind.as_operand_id()
            } else {
                let counter_kind = self.make_counter(|| Some(format!("{:?}->{:?}", from_bcb, to_bcb)));
                debug!("  Edge {:?}->{:?} gets a new counter: {}", from_bcb, to_bcb, self.format_counter(&counter_kind));
                self.basic_coverage_blocks[to_bcb].set_edge_counter_from(from_bcb, counter_kind)
            }
        } else {
            self.get_or_make_counter_operand(from_bcb, collect_intermediate_expressions)
        }
    }

    // loop through backedges

    // select inner loops before their outer loops, so the first matched loop for a given branch BCB
    // is it's inner-most loop

    // CASE #1:
    // if a branch_bcb is_dominated_by a loop bcb (target of backedge), and if any other branch_bcb is NOT dominated by the loop bcb,
    // add expression to the first branch_bcb that dominated by the loop bcb, and counters to all others. Compute expressions from
    // counter pairs as needed, to provide a single sum that can be subtracted from the branching block's counter.

    // CASE #2:
    // if all branch_bcb are dominated_by the loop bcb, no branch ends the loop (directly), so pick any branch to have the expression,

    // CASE #3:
    // if NONE of the branch_bcb are dominated_by the loop bcb, check if there's an outer loop (from stack of active loops?)
    // and re-do this check again to see if one of them jumps out of the outer loop while other(s) don't, and assign the expression
    // to one of the branch_bcb that is dominated_by that outer loop. (Continue this if none are dominated by the outer loop either.)

    // TODO(richkadel): In the last case above, also see the next TODO below. If all branches exit the loop then can we pass that info
    // to the predecessor (if only one??) so if the predecessor is a branch of another branching BCB, we know that the predecessor exits
    // the loop, and should have the counter, if the predecessor is in CASE #2 (none of the other branches of the predecessor's
    // branching BCB exited the loop?)

    // TODO(richkadel): What about a branch BCB that is another branching BCB, where both branches exit the loop?
    // Can I detect that somehow?

    // TODO(richkadel): For any node, N, and one of its successors, H (so N -> H), if (_also_)
    // N is_dominated_by H, then N -> H is a backedge. That is, we've identified that N -> H is
    // at least _one_ of possibly multiple arcs that loop back to the start of the loop with
    // "header" H, and this also means we've identified a loop, that has "header" H.
    //
    // H dominates everything inside the loop.
    //
    // TODO(richkadel): This doesn't make as much sense after renaming some terms:
    // So a branching BCB's branch in a BasicBlock that is_dominated_by H and has a branch to a
    // BasicBlock that is:
    //   * not H, and   ... (what if the branching BCB successor _is_ H? `continue`? is this a
    //     candidate for a middle or optional priority for getting a Counter?)
    //   * not is_dominated_by H
    // is a branch that jumps outside the loop, and should get an actual Counter, most likely
    //
    // Or perhaps conversely, a branching BCB is dominated by H with a branch that has a target that
    // ALSO is dominated by H should get a Expression.
    //
    //
    // So I need to identify all of the "H"'s, by identifying all of the backedges.
    //
    // If I have multiple H's (multiple loops), how do I decide which loop to compare a branch
    // BCB (by dominator) to?
    //
    // Can I assume the traversal order is helpful here? I.e., the "last" encountered loop
    // header is the (only?) one to compare to? (Probably not only... I don't see how that would
    // work for nested loops.)
    //
    // What about multiple loops in sequence?
    //
    //
    // What about nexted loops and jumping out of one or more of them at a time?

    fn make_counter<F>(&mut self, block_label_fn: F) -> CoverageKind
        where F: Fn() -> Option<String>
    {
        let counter = CoverageKind::Counter {
            function_source_hash: self.function_source_hash(),
            id: self.next_counter(),
        };
        if self.debug_counters.is_enabled() {
            self.debug_counters.add_counter(&counter, (block_label_fn)());
        }
        counter
    }

    fn make_expression<F>(
        &mut self,
        mut lhs: ExpressionOperandId,
        op: Op,
        mut rhs: ExpressionOperandId,
        block_label_fn: F,
    ) -> CoverageKind
        where F: Fn() -> Option<String>
    {
        if let Some(expressions_cache) = self.debug_expressions_cache.as_ref() {
            if let Some(CoverageKind::Expression { lhs: lhs_lhs, op, rhs: lhs_rhs, .. } ) = expressions_cache.get(&lhs) {
                if *lhs_rhs == ExpressionOperandId::ZERO {
                    lhs = *lhs_lhs;
                } else if *op == Op::Subtract && *lhs_rhs == rhs {
                    if let Some(lhs_expression) = expressions_cache.get(lhs_lhs) {
                        let expression = lhs_expression.clone();
                        return self.as_duplicate_expression(expression);
                    } else {
                        let counter = *lhs_lhs;
                        return self.make_identity_counter(counter);
                    }
                }
            }

            if let Some(CoverageKind::Expression { lhs: rhs_lhs, op, rhs: rhs_rhs, .. } ) = expressions_cache.get(&rhs) {
                if *rhs_rhs == ExpressionOperandId::ZERO {
                    rhs = *rhs_rhs;
                } else if *op == Op::Subtract && *rhs_rhs == lhs {
                    if let Some(rhs_expression) = expressions_cache.get(rhs_lhs) {
                        let expression = rhs_expression.clone();
                        return self.as_duplicate_expression(expression);
                    } else {
                        let counter = *rhs_lhs;
                        return self.make_identity_counter(counter);
                    }
                }
            }
        }

        let id = self.next_expression();
        let expression = CoverageKind::Expression { id, lhs, op, rhs };
        if let Some(expressions_cache) = self.debug_expressions_cache.as_mut() {
            expressions_cache.insert(id.into(), expression.clone());
        }
        if self.debug_counters.is_enabled() {
            self.debug_counters.add_counter(&expression, (block_label_fn)());
        }
        expression
    }

    fn as_duplicate_expression(&mut self, mut expression: CoverageKind) -> CoverageKind {
        let next_expression_id = if self.debug_expressions_cache.is_some() {
            Some(self.next_expression())
        } else {
            None
        };
        let expressions_cache = self.debug_expressions_cache.as_mut().expect("`as_duplicate_expression()` requires the debug_expressions_cache");
        match expression {
            CoverageKind::Expression { ref mut id, .. } => {
                *id = next_expression_id.expect("next_expression_id should be Some if there is a debug_expressions_cache");
                expressions_cache.insert(id.into(), expression.clone());
            }
            _ => bug!("make_duplicate_expression called with non-expression type: {:?}", expression),
        }
        expression
    }

    fn make_identity_counter(&mut self, counter_operand: ExpressionOperandId) -> CoverageKind {
        if let Some(expression) = self.debug_expressions_cache.as_ref().map_or(None, |c| c.get(&counter_operand)) {
            let new_expression = expression.clone();
            self.as_duplicate_expression(new_expression)
        } else {
            let some_block_label = if self.debug_counters.is_enabled() {
                self.debug_counters.some_block_label(counter_operand).cloned()
            } else {
                None
            };
            self.make_expression(counter_operand, Op::Add, ExpressionOperandId::ZERO, || some_block_label.clone())
        }
    }

    fn inject_statement(
        &mut self,
        counter_kind: CoverageKind,
        bb: BasicBlock,
        code_region: CodeRegion,
    ) {
        // TODO(richkadel): uncomment debug!
        // debug!("  injecting statement {:?} covering {:?}", counter_kind, code_region);

        let data = &mut self.mir_body[bb];
        let source_info = data.terminator().source_info;
        let statement = Statement {
            source_info,
            kind: StatementKind::Coverage(box Coverage { kind: counter_kind, code_region: Some(code_region) }),
        };
        data.statements.push(statement);
    }

    // Non-code expressions are injected into the coverage map, without generating executable code.
    fn inject_intermediate_expression(&mut self, expression: CoverageKind) {
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
            let tooltip = coverage_span.format_coverage_statements(tcx, self.mir_body);
            let CoverageSpan { span, bcb, .. } = coverage_span;
            let bcb_data = self.bcb_data(*bcb);
            let id = bcb_data.id();
            let leader_bb = bcb_data.leader_bb();
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

        let refinery = CoverageSpanRefinery::from_sorted_spans(initial_spans, &self.mir_body, &self.basic_coverage_blocks, &self.bcb_dominators);
        refinery.to_refined_spans()
    }
}

/// Converts the initial set of `CoverageSpan`s (one per MIR `Statement` or `Terminator`) into a
/// minimal set of `CoverageSpan`s, using the BCB CFG to determine where it is safe and useful to:
///
///  * Remove duplicate source code coverage regions
///  * Merge spans that represent continuous (both in source code and control flow), non-branching
///    execution
///  * Carve out (leave uncovered) any span that will be counted by another MIR (notably, closures)
struct CoverageSpanRefinery<'a, 'tcx> {

    /// The initial set of `CoverageSpan`s, sorted by `Span` (`lo` and `hi`) and by relative
    /// dominance between the `BasicCoverageBlock`s of equal `Span`s.
    sorted_spans_iter: std::vec::IntoIter<CoverageSpan>,

    /// The MIR, used to look up `BasicBlockData`.
    mir_body: &'a mir::Body<'tcx>,

    /// The BasicCoverageBlock Control Flow Graph (BCB CFG).
    basic_coverage_blocks: &'a BasicCoverageBlocks,

    /// The BCB CFG's dominators tree, used to compute the dominance relationships, if any.
    bcb_dominators: &'a Dominators<BasicCoverageBlock>,

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

impl<'a, 'tcx> CoverageSpanRefinery<'a, 'tcx> {
    fn from_sorted_spans(
        sorted_spans: Vec<CoverageSpan>,
        mir_body: &'a mir::Body<'tcx>,
        basic_coverage_blocks: &'a BasicCoverageBlocks,
        bcb_dominators: &'a Dominators<BasicCoverageBlock>,
    ) -> Self {
        let refined_spans = Vec::with_capacity(sorted_spans.len());
        let mut sorted_spans_iter = sorted_spans.into_iter();
        let prev = sorted_spans_iter.next().expect("at least one span");
        let prev_original_span = prev.span;
        Self {
            sorted_spans_iter,
            mir_body,
            basic_coverage_blocks,
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
        let CoverageSpanRefinery { mir_body, basic_coverage_blocks, pending_dups, mut refined_spans, .. } = self;
        for dup in pending_dups {
            debug!("    ...adding at least one pending dup={:?}", dup);
            refined_spans.push(dup);
        }
        refined_spans.push(prev);

        // Remove `CoverageSpan`s with empty spans ONLY if the empty `CoverageSpan`s BCB also has at
        // least one other non-empty `CoverageSpan`.
        let mut has_coverage = BitSet::new_empty(basic_coverage_blocks.num_nodes());
        for covspan in &refined_spans {
            if ! covspan.span.is_empty() {
                has_coverage.insert(covspan.bcb);
            }
        }
        refined_spans.retain(|covspan| {
            !(
                covspan.span.is_empty()
                && is_goto(&basic_coverage_blocks[covspan.bcb].terminator(mir_body).kind)
                && has_coverage.contains(covspan.bcb)
            )
        });

        // Remove `CoverageSpan`s derived from closures, originally added to ensure the coverage
        // regions for the current function leave room for the closure's own coverage regions
        // (injected separately, from the closure's own MIR).
        refined_spans.retain(|covspan| !covspan.is_closure);
        refined_spans
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



    // prev and any/all pending_dups are equal in non-dominance over the other.
    // So when comparing to prev, need to also compare to all pending dups.




// TODO(richkadel): UPDATE COMMENT IF WE ARE CHANGING THIS LOGIC!
    /// Called if `curr.span` = `prev.span` (and all `pending_dups` spans, if any).
    ///
    /// When two `CoverageSpan`s have the same `Span`, dominated spans can be discarded; but if
    /// neither `CoverageSpan` dominates the other, both (or possibly more than two) are held,
    /// until their disposition is determined. In this latter case, the `prev` dup is moved into
    /// `pending_dups` so the new `curr` dup can be moved to `prev` for the next iteration.
    fn hold_pending_dups_unless_dominated(&mut self) {
        // Equal coverage spans are ordered by dominators before dominated (if any), so it should be
        // impossible for `curr` to dominate any previous `CoverageSpan`.
        debug_assert!(!self.prev().is_dominated_by(self.curr(), self.bcb_dominators));

        let initial_pending_count = self.pending_dups.len();
        if initial_pending_count > 0 {
            let mut pending_dups = self.pending_dups.split_off(0);
            pending_dups.retain(|dup| !self.curr().is_dominated_by(dup, self.bcb_dominators));
            self.pending_dups.append(&mut pending_dups);
            if self.pending_dups.len() < initial_pending_count {
                debug!(
                    "  discarded {} of {} pending_dups that dominated curr",
                    initial_pending_count - self.pending_dups.len(),
                    initial_pending_count
                );
            }
        }

        if self.curr().is_dominated_by(self.prev(), self.bcb_dominators) {
            // Keep in mind, if discarding `prev`, that this method is called if
            // `curr.span` == `prev_original_span`, which may not be the full `prev.span` (if merged
            // during the previous iteration).
            debug!(
                "  different bcbs but SAME spans, and prev dominates curr. Discard prev={:?}",
                self.prev()
            );
            // TODO(richkadel): remove?
            // let discarded_prev = self.take_prev();
            self.cutoff_prev_at_overlapping_curr();
// TODO(richkadel): remove?
//        if self.curr().is_dominated_by(self.prev(), self.bcb_dominators) {
// IF WE KEEP IT, WE REALLY SHOULD CHECK ALL:
//        if self.pending_dups.iter().chain(self.some_prev.iter()).any(|dup|
//            self.curr().is_dominated_by(dup, self.bcb_dominators)) {

// TODO(richkadel): UPDATE COMMENT IF WE ARE CHANGING THIS LOGIC!
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

            // **** MAYBE UNCOMMENT, OR REMOVE IF NEW debug! above *** TODO(richkadel): uncomment debug!
            // debug!(
            //     "  different bcbs but SAME spans, and prev dominates curr. Drop curr and \
            //     keep prev for next iter. prev={:?}",
            //     self.prev()
            // );

// TODO(richkadel): Update logic and remove the following line if it works because this may have
// worked for the example above, but it seems to be consuming the `?` (try operator) as part of
// the call span that dominates it.

//            self.discard_curr();

// TODO(richkadel): If I don't change it, should I have checked not only prev dominates curr but
// (if not) also check any of the pending dups dominates curr?

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
// TODO(richkadel): Should Goto still be here? Delete if not
//        | TerminatorKind::Goto { .. }
        // For `FalseEdge`, only the `real` branch is taken, so it is similar to a `Goto`.
        | TerminatorKind::FalseEdge { .. } => None,

// TODO(richkadel): Add a comment if this works. Any `Goto` still in the BCB CFG is probably
// required, as an intermediate target for some conditional branches (typically, `SwitchInt`
// targets). Some of these `Goto` branch blocks (BCB and BB) have multiple incoming edges (that is,
// not just a single incoming edge from a SwitchInt), so they are often required to accurately
// represent the control flow.
//
// Since these retained `Goto` terminators generally impact control flow, they often generate their
// own `BasicCoverageBlock`, but their `source_info.span` typically covers the source code for an
// entire branch (that might or might not be executed, depending on the condition that may (or may
// not) lead to the `Goto` branch. This source span is far too broad. For `Goto` terminators that
// are part of a BCB with other statements/terminators with usable `Span`s, ignoring the `Goto`
// `Span` is OK, because the other statements/terminators in the BCB generate `CoverageSpans`
// that represent the code executed leading up to (and including) that `Goto`.
//
// But for `Goto` terminators without other statements/terminators, there needs to be some visible
// region to count, that doesn't include code that was not executed. So in this case, we set the
// `Span` to the 0-length span from and to `source_info.span.hi()`.
//
// At least one character will be covered--the character immediately following the `Span` position.
// (`make_code_region()` ensures all zero-length spans are extended to at least one character.)
//
// Note that, if we did not select some non-zero length `Span` at this location, but the `Goto`s
// BCB still needed to be counted (other `CoverageSpan`-dependent `Expression`s may depend on the
// execution count for this `Goto`), we have at least two other less-appealing choices:
//   1. Insert a truly zero-length span on some line. In this case, there may be no visible
//      coverage region, but its line is still counted, which is more confusing.
//   2. Insert a span for a line beyond the end of the file. This works well for `llvm-cov show`
//      reports, but `llvm-cov export` reports still count the lines and regions outside the
//      file's line range.
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

// TODO(richkadel): Update this function comment.
// It only works to use an empty span on an actual source line if the line already has
// a coverage code region. It works because
//   compiler/rustc_codegen_llvm/src/coverageinfo/mapgen.rs
// knows to look for empty spans, and generate a `GapRegion` instead of a `CodeRegion`.
//
/// Make a non-reportable code region
/// Count empty spans, but don't make them reportable as coverage. Set the source
/// position out of range. (Note that `llvm-cov` fails to report coverage if any
/// coverage region starts line `0`, so we are using a line number just beyond
/// the last line of the file.)
fn make_non_reportable_code_region(
    file_name: Symbol,
//    last_line: usize,
//    last_col: CharPos,
    source_file: &Lrc<SourceFile>,
    span: Span,
) -> CodeRegion {
    // let line_past_end_of_file = (source_file.lines.len() + 1) as u32;
    // CodeRegion {
    //     file_name,
    //     start_line: line_past_end_of_file,
    //     start_col: 1,
    //     end_line: line_past_end_of_file,
    //     end_col: 1,
    // }
    // CodeRegion {
    //     file_name,
    //     start_line: last_line as u32,
    //     start_col: last_col.to_u32() + 1,
    //     end_line: last_line as u32,
    //     end_col: last_col.to_u32() + 1,
    // }
    let (start_line, start_col) = source_file.lookup_file_pos(span.hi());
    let (end_line, end_col) = (start_line, start_col);
    CodeRegion {
        file_name,
        start_line: start_line as u32,
        start_col: start_col.to_u32() + 1,
        end_line: end_line as u32,
        end_col: end_col.to_u32() + 1,
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

#[inline(always)]
fn is_goto(term_kind: &TerminatorKind<'tcx>) -> bool {
    match term_kind {
        TerminatorKind::Goto { .. } => true,
        _ => false,
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

fn bcb_filtered_successors<'a, 'tcx>(body: &'tcx &'a mir::Body<'tcx>, term_kind: &'tcx TerminatorKind<'tcx>) -> Box<dyn Iterator<Item = &'a BasicBlock> + 'a> {
    let mut successors = term_kind.successors();
    box match &term_kind {
        // SwitchInt successors are never unwind, and all of them should be traversed.
        TerminatorKind::SwitchInt { .. } => successors,
        // For all other kinds, return only the first successor, if any, and ignore unwinds.
        // NOTE: `chain(&[])` is required to coerce the `option::iter` (from
        // `next().into_iter()`) into the `mir::Successors` aliased type.
        _ => successors.next().into_iter().chain(&[]),
    }.filter(move |&&successor| body[successor].terminator().kind != TerminatorKind::Unreachable)
}

pub struct ShortCircuitPreorder<
    'a,
    'tcx,
    F: Fn(&'tcx &'a mir::Body<'tcx>, &'tcx TerminatorKind<'tcx>) -> Box<dyn Iterator<Item = &'a BasicBlock> + 'a>,
> {
    body: &'tcx &'a mir::Body<'tcx>,
    visited: BitSet<BasicBlock>,
    worklist: Vec<BasicBlock>,
    filtered_successors: F,
}

impl<'a, 'tcx, F: Fn(&'tcx &'a mir::Body<'tcx>, &'tcx TerminatorKind<'tcx>) -> Box<dyn Iterator<Item = &'a BasicBlock> + 'a>>
    ShortCircuitPreorder<'a, 'tcx, F>
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

impl<'a: 'tcx, 'tcx, F: Fn(&'tcx &'a mir::Body<'tcx>, &'tcx TerminatorKind<'tcx>) -> Box<dyn Iterator<Item = &'a BasicBlock> + 'a>> Iterator
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

// TODO(richkadel): try_error_result.rs
// When executing the Result as Try>::from_error() returns to one or more
// Goto that then targets Return.
// Both the Goto (after error) and the Return have coverage at the last
// bytepos, 0-length Span.
//
// How should I eliminate one, and which one, to avoid counting both.
