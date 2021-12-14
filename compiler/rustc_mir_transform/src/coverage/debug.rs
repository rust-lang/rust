//! The `InstrumentCoverage` MIR pass implementation includes debugging tools and options
//! to help developers understand and/or improve the analysis and instrumentation of a MIR.
//!
//! To enable coverage, include the rustc command line option:
//!
//!   * `-Z instrument-coverage`
//!
//! MIR Dump Files, with additional `CoverageGraph` graphviz and `CoverageSpan` spanview
//! ------------------------------------------------------------------------------------
//!
//! Additional debugging options include:
//!
//!   * `-Z dump-mir=InstrumentCoverage` - Generate `.mir` files showing the state of the MIR,
//!     before and after the `InstrumentCoverage` pass, for each compiled function.
//!
//!   * `-Z dump-mir-graphviz` - If `-Z dump-mir` is also enabled for the current MIR node path,
//!     each MIR dump is accompanied by a before-and-after graphical view of the MIR, in Graphviz
//!     `.dot` file format (which can be visually rendered as a graph using any of a number of free
//!     Graphviz viewers and IDE extensions).
//!
//!     For the `InstrumentCoverage` pass, this option also enables generation of an additional
//!     Graphviz `.dot` file for each function, rendering the `CoverageGraph`: the control flow
//!     graph (CFG) of `BasicCoverageBlocks` (BCBs), as nodes, internally labeled to show the
//!     `CoverageSpan`-based MIR elements each BCB represents (`BasicBlock`s, `Statement`s and
//!     `Terminator`s), assigned coverage counters and/or expressions, and edge counters, as needed.
//!
//!     (Note the additional option, `-Z graphviz-dark-mode`, can be added, to change the rendered
//!     output from its default black-on-white background to a dark color theme, if desired.)
//!
//!   * `-Z dump-mir-spanview` - If `-Z dump-mir` is also enabled for the current MIR node path,
//!     each MIR dump is accompanied by a before-and-after `.html` document showing the function's
//!     original source code, highlighted by it's MIR spans, at the `statement`-level (by default),
//!     `terminator` only, or encompassing span for the `Terminator` plus all `Statement`s, in each
//!     `block` (`BasicBlock`).
//!
//!     For the `InstrumentCoverage` pass, this option also enables generation of an additional
//!     spanview `.html` file for each function, showing the aggregated `CoverageSpan`s that will
//!     require counters (or counter expressions) for accurate coverage analysis.
//!
//! Debug Logging
//! -------------
//!
//! The `InstrumentCoverage` pass includes debug logging messages at various phases and decision
//! points, which can be enabled via environment variable:
//!
//! ```shell
//! RUSTC_LOG=rustc_mir_transform::transform::coverage=debug
//! ```
//!
//! Other module paths with coverage-related debug logs may also be of interest, particularly for
//! debugging the coverage map data, injected as global variables in the LLVM IR (during rustc's
//! code generation pass). For example:
//!
//! ```shell
//! RUSTC_LOG=rustc_mir_transform::transform::coverage,rustc_codegen_ssa::coverageinfo,rustc_codegen_llvm::coverageinfo=debug
//! ```
//!
//! Coverage Debug Options
//! ---------------------------------
//!
//! Additional debugging options can be enabled using the environment variable:
//!
//! ```shell
//! RUSTC_COVERAGE_DEBUG_OPTIONS=<options>
//! ```
//!
//! These options are comma-separated, and specified in the format `option-name=value`. For example:
//!
//! ```shell
//! $ RUSTC_COVERAGE_DEBUG_OPTIONS=counter-format=id+operation,allow-unused-expressions=yes cargo build
//! ```
//!
//! Coverage debug options include:
//!
//!   * `allow-unused-expressions=yes` or `no` (default: `no`)
//!
//!     The `InstrumentCoverage` algorithms _should_ only create and assign expressions to a
//!     `BasicCoverageBlock`, or an incoming edge, if that expression is either (a) required to
//!     count a `CoverageSpan`, or (b) a dependency of some other required counter expression.
//!
//!     If an expression is generated that does not map to a `CoverageSpan` or dependency, this
//!     probably indicates there was a bug in the algorithm that creates and assigns counters
//!     and expressions.
//!
//!     When this kind of bug is encountered, the rustc compiler will panic by default. Setting:
//!     `allow-unused-expressions=yes` will log a warning message instead of panicking (effectively
//!     ignoring the unused expressions), which may be helpful when debugging the root cause of
//!     the problem.
//!
//!   * `counter-format=<choices>`, where `<choices>` can be any plus-separated combination of `id`,
//!     `block`, and/or `operation` (default: `block+operation`)
//!
//!     This option effects both the `CoverageGraph` (graphviz `.dot` files) and debug logging, when
//!     generating labels for counters and expressions.
//!
//!     Depending on the values and combinations, counters can be labeled by:
//!
//!       * `id` - counter or expression ID (ascending counter IDs, starting at 1, or descending
//!         expression IDs, starting at `u32:MAX`)
//!       * `block` - the `BasicCoverageBlock` label (for example, `bcb0`) or edge label (for
//!         example `bcb0->bcb1`), for counters or expressions assigned to count a
//!         `BasicCoverageBlock` or edge. Intermediate expressions (not directly associated with
//!         a BCB or edge) will be labeled by their expression ID, unless `operation` is also
//!         specified.
//!       * `operation` - applied to expressions only, labels include the left-hand-side counter
//!         or expression label (lhs operand), the operator (`+` or `-`), and the right-hand-side
//!         counter or expression (rhs operand). Expression operand labels are generated
//!         recursively, generating labels with nested operations, enclosed in parentheses
//!         (for example: `bcb2 + (bcb0 - bcb1)`).

use super::graph::{BasicCoverageBlock, BasicCoverageBlockData, CoverageGraph};
use super::spans::CoverageSpan;

use rustc_middle::mir::create_dump_file;
use rustc_middle::mir::generic_graphviz::GraphvizWriter;
use rustc_middle::mir::spanview::{self, SpanViewable};

use rustc_data_structures::fx::FxHashMap;
use rustc_middle::mir::coverage::*;
use rustc_middle::mir::{self, BasicBlock, TerminatorKind};
use rustc_middle::ty::TyCtxt;
use rustc_span::Span;

use std::iter;
use std::lazy::SyncOnceCell;

pub const NESTED_INDENT: &str = "    ";

const RUSTC_COVERAGE_DEBUG_OPTIONS: &str = "RUSTC_COVERAGE_DEBUG_OPTIONS";

pub(super) fn debug_options<'a>() -> &'a DebugOptions {
    static DEBUG_OPTIONS: SyncOnceCell<DebugOptions> = SyncOnceCell::new();

    &DEBUG_OPTIONS.get_or_init(DebugOptions::from_env)
}

/// Parses and maintains coverage-specific debug options captured from the environment variable
/// "RUSTC_COVERAGE_DEBUG_OPTIONS", if set.
#[derive(Debug, Clone)]
pub(super) struct DebugOptions {
    pub allow_unused_expressions: bool,
    counter_format: ExpressionFormat,
}

impl DebugOptions {
    fn from_env() -> Self {
        let mut allow_unused_expressions = true;
        let mut counter_format = ExpressionFormat::default();

        if let Ok(env_debug_options) = std::env::var(RUSTC_COVERAGE_DEBUG_OPTIONS) {
            for setting_str in env_debug_options.replace(' ', "").replace('-', "_").split(',') {
                let (option, value) = match setting_str.split_once('=') {
                    None => (setting_str, None),
                    Some((k, v)) => (k, Some(v)),
                };
                match option {
                    "allow_unused_expressions" => {
                        allow_unused_expressions = bool_option_val(option, value);
                        debug!(
                            "{} env option `allow_unused_expressions` is set to {}",
                            RUSTC_COVERAGE_DEBUG_OPTIONS, allow_unused_expressions
                        );
                    }
                    "counter_format" => {
                        match value {
                            None => {
                                bug!(
                                    "`{}` option in environment variable {} requires one or more \
                                    plus-separated choices (a non-empty subset of \
                                    `id+block+operation`)",
                                    option,
                                    RUSTC_COVERAGE_DEBUG_OPTIONS
                                );
                            }
                            Some(val) => {
                                counter_format = counter_format_option_val(val);
                                debug!(
                                    "{} env option `counter_format` is set to {:?}",
                                    RUSTC_COVERAGE_DEBUG_OPTIONS, counter_format
                                );
                            }
                        };
                    }
                    _ => bug!(
                        "Unsupported setting `{}` in environment variable {}",
                        option,
                        RUSTC_COVERAGE_DEBUG_OPTIONS
                    ),
                };
            }
        }

        Self { allow_unused_expressions, counter_format }
    }
}

fn bool_option_val(option: &str, some_strval: Option<&str>) -> bool {
    if let Some(val) = some_strval {
        if vec!["yes", "y", "on", "true"].contains(&val) {
            true
        } else if vec!["no", "n", "off", "false"].contains(&val) {
            false
        } else {
            bug!(
                "Unsupported value `{}` for option `{}` in environment variable {}",
                option,
                val,
                RUSTC_COVERAGE_DEBUG_OPTIONS
            )
        }
    } else {
        true
    }
}

fn counter_format_option_val(strval: &str) -> ExpressionFormat {
    let mut counter_format = ExpressionFormat { id: false, block: false, operation: false };
    let components = strval.splitn(3, '+');
    for component in components {
        match component {
            "id" => counter_format.id = true,
            "block" => counter_format.block = true,
            "operation" => counter_format.operation = true,
            _ => bug!(
                "Unsupported counter_format choice `{}` in environment variable {}",
                component,
                RUSTC_COVERAGE_DEBUG_OPTIONS
            ),
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
        Self { id: false, block: true, operation: true }
    }
}

/// If enabled, this struct maintains a map from `CoverageKind` IDs (as `ExpressionOperandId`) to
/// the `CoverageKind` data and optional label (normally, the counter's associated
/// `BasicCoverageBlock` format string, if any).
///
/// Use `format_counter` to convert one of these `CoverageKind` counters to a debug output string,
/// as directed by the `DebugOptions`. This allows the format of counter labels in logs and dump
/// files (including the `CoverageGraph` graphviz file) to be changed at runtime, via environment
/// variable.
///
/// `DebugCounters` supports a recursive rendering of `Expression` counters, so they can be
/// presented as nested expressions such as `(bcb3 - (bcb0 + bcb1))`.
pub(super) struct DebugCounters {
    some_counters: Option<FxHashMap<ExpressionOperandId, DebugCounter>>,
}

impl DebugCounters {
    pub fn new() -> Self {
        Self { some_counters: None }
    }

    pub fn enable(&mut self) {
        debug_assert!(!self.is_enabled());
        self.some_counters.replace(FxHashMap::default());
    }

    pub fn is_enabled(&self) -> bool {
        self.some_counters.is_some()
    }

    pub fn add_counter(&mut self, counter_kind: &CoverageKind, some_block_label: Option<String>) {
        if let Some(counters) = &mut self.some_counters {
            let id: ExpressionOperandId = match *counter_kind {
                CoverageKind::Counter { id, .. } => id.into(),
                CoverageKind::Expression { id, .. } => id.into(),
                _ => bug!(
                    "the given `CoverageKind` is not an counter or expression: {:?}",
                    counter_kind
                ),
            };
            counters
                .try_insert(id, DebugCounter::new(counter_kind.clone(), some_block_label))
                .expect("attempt to add the same counter_kind to DebugCounters more than once");
        }
    }

    pub fn some_block_label(&self, operand: ExpressionOperandId) -> Option<&String> {
        self.some_counters.as_ref().map_or(None, |counters| {
            counters
                .get(&operand)
                .map_or(None, |debug_counter| debug_counter.some_block_label.as_ref())
        })
    }

    pub fn format_counter(&self, counter_kind: &CoverageKind) -> String {
        match *counter_kind {
            CoverageKind::Counter { .. } => {
                format!("Counter({})", self.format_counter_kind(counter_kind))
            }
            CoverageKind::Expression { .. } => {
                format!("Expression({})", self.format_counter_kind(counter_kind))
            }
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
                        format!("#{} = ", id.index())
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
            CoverageKind::Expression { id, .. } => id.into(),
            _ => {
                bug!("the given `CoverageKind` is not an counter or expression: {:?}", counter_kind)
            }
        };
        if self.some_counters.is_some() && (counter_format.block || !counter_format.id) {
            let counters = self.some_counters.as_ref().unwrap();
            if let Some(DebugCounter { some_block_label: Some(block_label), .. }) =
                counters.get(&id)
            {
                return if counter_format.id {
                    format!("{}#{}", block_label, id.index())
                } else {
                    block_label.to_string()
                };
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
                            return format!(
                                "{}:({})",
                                block_label,
                                self.format_counter_kind(counter_kind)
                            );
                        }
                    }
                    return format!("({})", self.format_counter_kind(counter_kind));
                }
                return self.format_counter_kind(counter_kind);
            }
        }
        format!("#{}", operand.index())
    }
}

/// A non-public support class to `DebugCounters`.
#[derive(Debug)]
struct DebugCounter {
    counter_kind: CoverageKind,
    some_block_label: Option<String>,
}

impl DebugCounter {
    fn new(counter_kind: CoverageKind, some_block_label: Option<String>) -> Self {
        Self { counter_kind, some_block_label }
    }
}

/// If enabled, this data structure captures additional debugging information used when generating
/// a Graphviz (.dot file) representation of the `CoverageGraph`, for debugging purposes.
pub(super) struct GraphvizData {
    some_bcb_to_coverage_spans_with_counters:
        Option<FxHashMap<BasicCoverageBlock, Vec<(CoverageSpan, CoverageKind)>>>,
    some_bcb_to_dependency_counters: Option<FxHashMap<BasicCoverageBlock, Vec<CoverageKind>>>,
    some_edge_to_counter: Option<FxHashMap<(BasicCoverageBlock, BasicBlock), CoverageKind>>,
}

impl GraphvizData {
    pub fn new() -> Self {
        Self {
            some_bcb_to_coverage_spans_with_counters: None,
            some_bcb_to_dependency_counters: None,
            some_edge_to_counter: None,
        }
    }

    pub fn enable(&mut self) {
        debug_assert!(!self.is_enabled());
        self.some_bcb_to_coverage_spans_with_counters = Some(FxHashMap::default());
        self.some_bcb_to_dependency_counters = Some(FxHashMap::default());
        self.some_edge_to_counter = Some(FxHashMap::default());
    }

    pub fn is_enabled(&self) -> bool {
        self.some_bcb_to_coverage_spans_with_counters.is_some()
    }

    pub fn add_bcb_coverage_span_with_counter(
        &mut self,
        bcb: BasicCoverageBlock,
        coverage_span: &CoverageSpan,
        counter_kind: &CoverageKind,
    ) {
        if let Some(bcb_to_coverage_spans_with_counters) =
            self.some_bcb_to_coverage_spans_with_counters.as_mut()
        {
            bcb_to_coverage_spans_with_counters
                .entry(bcb)
                .or_insert_with(Vec::new)
                .push((coverage_span.clone(), counter_kind.clone()));
        }
    }

    pub fn get_bcb_coverage_spans_with_counters(
        &self,
        bcb: BasicCoverageBlock,
    ) -> Option<&Vec<(CoverageSpan, CoverageKind)>> {
        if let Some(bcb_to_coverage_spans_with_counters) =
            self.some_bcb_to_coverage_spans_with_counters.as_ref()
        {
            bcb_to_coverage_spans_with_counters.get(&bcb)
        } else {
            None
        }
    }

    pub fn add_bcb_dependency_counter(
        &mut self,
        bcb: BasicCoverageBlock,
        counter_kind: &CoverageKind,
    ) {
        if let Some(bcb_to_dependency_counters) = self.some_bcb_to_dependency_counters.as_mut() {
            bcb_to_dependency_counters
                .entry(bcb)
                .or_insert_with(Vec::new)
                .push(counter_kind.clone());
        }
    }

    pub fn get_bcb_dependency_counters(
        &self,
        bcb: BasicCoverageBlock,
    ) -> Option<&Vec<CoverageKind>> {
        if let Some(bcb_to_dependency_counters) = self.some_bcb_to_dependency_counters.as_ref() {
            bcb_to_dependency_counters.get(&bcb)
        } else {
            None
        }
    }

    pub fn set_edge_counter(
        &mut self,
        from_bcb: BasicCoverageBlock,
        to_bb: BasicBlock,
        counter_kind: &CoverageKind,
    ) {
        if let Some(edge_to_counter) = self.some_edge_to_counter.as_mut() {
            edge_to_counter
                .try_insert((from_bcb, to_bb), counter_kind.clone())
                .expect("invalid attempt to insert more than one edge counter for the same edge");
        }
    }

    pub fn get_edge_counter(
        &self,
        from_bcb: BasicCoverageBlock,
        to_bb: BasicBlock,
    ) -> Option<&CoverageKind> {
        if let Some(edge_to_counter) = self.some_edge_to_counter.as_ref() {
            edge_to_counter.get(&(from_bcb, to_bb))
        } else {
            None
        }
    }
}

/// If enabled, this struct captures additional data used to track whether expressions were used,
/// directly or indirectly, to compute the coverage counts for all `CoverageSpan`s, and any that are
/// _not_ used are retained in the `unused_expressions` Vec, to be included in debug output (logs
/// and/or a `CoverageGraph` graphviz output).
pub(super) struct UsedExpressions {
    some_used_expression_operands:
        Option<FxHashMap<ExpressionOperandId, Vec<InjectedExpressionId>>>,
    some_unused_expressions:
        Option<Vec<(CoverageKind, Option<BasicCoverageBlock>, BasicCoverageBlock)>>,
}

impl UsedExpressions {
    pub fn new() -> Self {
        Self { some_used_expression_operands: None, some_unused_expressions: None }
    }

    pub fn enable(&mut self) {
        debug_assert!(!self.is_enabled());
        self.some_used_expression_operands = Some(FxHashMap::default());
        self.some_unused_expressions = Some(Vec::new());
    }

    pub fn is_enabled(&self) -> bool {
        self.some_used_expression_operands.is_some()
    }

    pub fn add_expression_operands(&mut self, expression: &CoverageKind) {
        if let Some(used_expression_operands) = self.some_used_expression_operands.as_mut() {
            if let CoverageKind::Expression { id, lhs, rhs, .. } = *expression {
                used_expression_operands.entry(lhs).or_insert_with(Vec::new).push(id);
                used_expression_operands.entry(rhs).or_insert_with(Vec::new).push(id);
            }
        }
    }

    pub fn expression_is_used(&self, expression: &CoverageKind) -> bool {
        if let Some(used_expression_operands) = self.some_used_expression_operands.as_ref() {
            used_expression_operands.contains_key(&expression.as_operand_id())
        } else {
            false
        }
    }

    pub fn add_unused_expression_if_not_found(
        &mut self,
        expression: &CoverageKind,
        edge_from_bcb: Option<BasicCoverageBlock>,
        target_bcb: BasicCoverageBlock,
    ) {
        if let Some(used_expression_operands) = self.some_used_expression_operands.as_ref() {
            if !used_expression_operands.contains_key(&expression.as_operand_id()) {
                self.some_unused_expressions.as_mut().unwrap().push((
                    expression.clone(),
                    edge_from_bcb,
                    target_bcb,
                ));
            }
        }
    }

    /// Return the list of unused counters (if any) as a tuple with the counter (`CoverageKind`),
    /// optional `from_bcb` (if it was an edge counter), and `target_bcb`.
    pub fn get_unused_expressions(
        &self,
    ) -> Vec<(CoverageKind, Option<BasicCoverageBlock>, BasicCoverageBlock)> {
        if let Some(unused_expressions) = self.some_unused_expressions.as_ref() {
            unused_expressions.clone()
        } else {
            Vec::new()
        }
    }

    /// If enabled, validate that every BCB or edge counter not directly associated with a coverage
    /// span is at least indirectly associated (it is a dependency of a BCB counter that _is_
    /// associated with a coverage span).
    pub fn validate(
        &mut self,
        bcb_counters_without_direct_coverage_spans: &Vec<(
            Option<BasicCoverageBlock>,
            BasicCoverageBlock,
            CoverageKind,
        )>,
    ) {
        if self.is_enabled() {
            let mut not_validated = bcb_counters_without_direct_coverage_spans
                .iter()
                .map(|(_, _, counter_kind)| counter_kind)
                .collect::<Vec<_>>();
            let mut validating_count = 0;
            while not_validated.len() != validating_count {
                let to_validate = not_validated.split_off(0);
                validating_count = to_validate.len();
                for counter_kind in to_validate {
                    if self.expression_is_used(counter_kind) {
                        self.add_expression_operands(counter_kind);
                    } else {
                        not_validated.push(counter_kind);
                    }
                }
            }
        }
    }

    pub fn alert_on_unused_expressions(&self, debug_counters: &DebugCounters) {
        if let Some(unused_expressions) = self.some_unused_expressions.as_ref() {
            for (counter_kind, edge_from_bcb, target_bcb) in unused_expressions {
                let unused_counter_message = if let Some(from_bcb) = edge_from_bcb.as_ref() {
                    format!(
                        "non-coverage edge counter found without a dependent expression, in \
                        {:?}->{:?}; counter={}",
                        from_bcb,
                        target_bcb,
                        debug_counters.format_counter(&counter_kind),
                    )
                } else {
                    format!(
                        "non-coverage counter found without a dependent expression, in {:?}; \
                        counter={}",
                        target_bcb,
                        debug_counters.format_counter(&counter_kind),
                    )
                };

                if debug_options().allow_unused_expressions {
                    debug!("WARNING: {}", unused_counter_message);
                } else {
                    bug!("{}", unused_counter_message);
                }
            }
        }
    }
}

/// Generates the MIR pass `CoverageSpan`-specific spanview dump file.
pub(super) fn dump_coverage_spanview<'tcx>(
    tcx: TyCtxt<'tcx>,
    mir_body: &mir::Body<'tcx>,
    basic_coverage_blocks: &CoverageGraph,
    pass_name: &str,
    body_span: Span,
    coverage_spans: &Vec<CoverageSpan>,
) {
    let mir_source = mir_body.source;
    let def_id = mir_source.def_id();

    let span_viewables = span_viewables(tcx, mir_body, basic_coverage_blocks, &coverage_spans);
    let mut file = create_dump_file(tcx, "html", None, pass_name, &0, mir_source)
        .expect("Unexpected error creating MIR spanview HTML file");
    let crate_name = tcx.crate_name(def_id.krate);
    let item_name = tcx.def_path(def_id).to_filename_friendly_no_crate();
    let title = format!("{}.{} - Coverage Spans", crate_name, item_name);
    spanview::write_document(tcx, body_span, span_viewables, &title, &mut file)
        .expect("Unexpected IO error dumping coverage spans as HTML");
}

/// Converts the computed `BasicCoverageBlockData`s into `SpanViewable`s.
fn span_viewables<'tcx>(
    tcx: TyCtxt<'tcx>,
    mir_body: &mir::Body<'tcx>,
    basic_coverage_blocks: &CoverageGraph,
    coverage_spans: &Vec<CoverageSpan>,
) -> Vec<SpanViewable> {
    let mut span_viewables = Vec::new();
    for coverage_span in coverage_spans {
        let tooltip = coverage_span.format_coverage_statements(tcx, mir_body);
        let CoverageSpan { span, bcb, .. } = coverage_span;
        let bcb_data = &basic_coverage_blocks[*bcb];
        let id = bcb_data.id();
        let leader_bb = bcb_data.leader_bb();
        span_viewables.push(SpanViewable { bb: leader_bb, span: *span, id, tooltip });
    }
    span_viewables
}

/// Generates the MIR pass coverage-specific graphviz dump file.
pub(super) fn dump_coverage_graphviz<'tcx>(
    tcx: TyCtxt<'tcx>,
    mir_body: &mir::Body<'tcx>,
    pass_name: &str,
    basic_coverage_blocks: &CoverageGraph,
    debug_counters: &DebugCounters,
    graphviz_data: &GraphvizData,
    intermediate_expressions: &Vec<CoverageKind>,
    debug_used_expressions: &UsedExpressions,
) {
    let mir_source = mir_body.source;
    let def_id = mir_source.def_id();
    let node_content = |bcb| {
        bcb_to_string_sections(
            tcx,
            mir_body,
            debug_counters,
            &basic_coverage_blocks[bcb],
            graphviz_data.get_bcb_coverage_spans_with_counters(bcb),
            graphviz_data.get_bcb_dependency_counters(bcb),
            // intermediate_expressions are injected into the mir::START_BLOCK, so
            // include them in the first BCB.
            if bcb.index() == 0 { Some(&intermediate_expressions) } else { None },
        )
    };
    let edge_labels = |from_bcb| {
        let from_bcb_data = &basic_coverage_blocks[from_bcb];
        let from_terminator = from_bcb_data.terminator(mir_body);
        let mut edge_labels = from_terminator.kind.fmt_successor_labels();
        edge_labels.retain(|label| label != "unreachable");
        let edge_counters = from_terminator
            .successors()
            .map(|&successor_bb| graphviz_data.get_edge_counter(from_bcb, successor_bb));
        iter::zip(&edge_labels, edge_counters)
            .map(|(label, some_counter)| {
                if let Some(counter) = some_counter {
                    format!("{}\n{}", label, debug_counters.format_counter(counter))
                } else {
                    label.to_string()
                }
            })
            .collect::<Vec<_>>()
    };
    let graphviz_name = format!("Cov_{}_{}", def_id.krate.index(), def_id.index.index());
    let mut graphviz_writer =
        GraphvizWriter::new(basic_coverage_blocks, &graphviz_name, node_content, edge_labels);
    let unused_expressions = debug_used_expressions.get_unused_expressions();
    if unused_expressions.len() > 0 {
        graphviz_writer.set_graph_label(&format!(
            "Unused expressions:\n  {}",
            unused_expressions
                .as_slice()
                .iter()
                .map(|(counter_kind, edge_from_bcb, target_bcb)| {
                    if let Some(from_bcb) = edge_from_bcb.as_ref() {
                        format!(
                            "{:?}->{:?}: {}",
                            from_bcb,
                            target_bcb,
                            debug_counters.format_counter(&counter_kind),
                        )
                    } else {
                        format!(
                            "{:?}: {}",
                            target_bcb,
                            debug_counters.format_counter(&counter_kind),
                        )
                    }
                })
                .collect::<Vec<_>>()
                .join("\n  ")
        ));
    }
    let mut file = create_dump_file(tcx, "dot", None, pass_name, &0, mir_source)
        .expect("Unexpected error creating BasicCoverageBlock graphviz DOT file");
    graphviz_writer
        .write_graphviz(tcx, &mut file)
        .expect("Unexpected error writing BasicCoverageBlock graphviz DOT file");
}

fn bcb_to_string_sections<'tcx>(
    tcx: TyCtxt<'tcx>,
    mir_body: &mir::Body<'tcx>,
    debug_counters: &DebugCounters,
    bcb_data: &BasicCoverageBlockData,
    some_coverage_spans_with_counters: Option<&Vec<(CoverageSpan, CoverageKind)>>,
    some_dependency_counters: Option<&Vec<CoverageKind>>,
    some_intermediate_expressions: Option<&Vec<CoverageKind>>,
) -> Vec<String> {
    let len = bcb_data.basic_blocks.len();
    let mut sections = Vec::new();
    if let Some(collect_intermediate_expressions) = some_intermediate_expressions {
        sections.push(
            collect_intermediate_expressions
                .iter()
                .map(|expression| {
                    format!("Intermediate {}", debug_counters.format_counter(expression))
                })
                .collect::<Vec<_>>()
                .join("\n"),
        );
    }
    if let Some(coverage_spans_with_counters) = some_coverage_spans_with_counters {
        sections.push(
            coverage_spans_with_counters
                .iter()
                .map(|(covspan, counter)| {
                    format!(
                        "{} at {}",
                        debug_counters.format_counter(counter),
                        covspan.format(tcx, mir_body)
                    )
                })
                .collect::<Vec<_>>()
                .join("\n"),
        );
    }
    if let Some(dependency_counters) = some_dependency_counters {
        sections.push(format!(
            "Non-coverage counters:\n  {}",
            dependency_counters
                .iter()
                .map(|counter| debug_counters.format_counter(counter))
                .collect::<Vec<_>>()
                .join("  \n"),
        ));
    }
    if let Some(counter_kind) = &bcb_data.counter_kind {
        sections.push(format!("{:?}", counter_kind));
    }
    let non_term_blocks = bcb_data.basic_blocks[0..len - 1]
        .iter()
        .map(|&bb| format!("{:?}: {}", bb, term_type(&mir_body[bb].terminator().kind)))
        .collect::<Vec<_>>();
    if non_term_blocks.len() > 0 {
        sections.push(non_term_blocks.join("\n"));
    }
    sections.push(format!(
        "{:?}: {}",
        bcb_data.basic_blocks.last().unwrap(),
        term_type(&bcb_data.terminator(mir_body).kind)
    ));
    sections
}

/// Returns a simple string representation of a `TerminatorKind` variant, independent of any
/// values it might hold.
pub(super) fn term_type(kind: &TerminatorKind<'_>) -> &'static str {
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
