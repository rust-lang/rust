use super::graph::{BasicCoverageBlock, BasicCoverageBlockData, CoverageGraph};
use super::spans::CoverageSpan;

use crate::util::generic_graphviz::GraphvizWriter;
use crate::util::pretty;
use crate::util::spanview::{self, SpanViewable};

use rustc_data_structures::fx::FxHashMap;
use rustc_index::vec::Idx;
use rustc_middle::mir::coverage::*;
use rustc_middle::mir::{self, BasicBlock, TerminatorKind};
use rustc_middle::ty::TyCtxt;

use std::lazy::SyncOnceCell;

const RUSTC_COVERAGE_DEBUG_OPTIONS: &str = "RUSTC_COVERAGE_DEBUG_OPTIONS";

pub(crate) fn debug_options<'a>() -> &'a DebugOptions {
    static DEBUG_OPTIONS: SyncOnceCell<DebugOptions> = SyncOnceCell::new();

    &DEBUG_OPTIONS.get_or_init(|| DebugOptions::new())
}

/// Parses and maintains coverage-specific debug options captured from the environment variable
/// "RUSTC_COVERAGE_DEBUG_OPTIONS", if set. Options can be set on the command line by, for example:
///
///     $ RUSTC_COVERAGE_DEBUG_OPTIONS=counter-format=block cargo build
#[derive(Debug, Clone)]
pub(crate) struct DebugOptions {
    counter_format: ExpressionFormat,
}

impl DebugOptions {
    fn new() -> Self {
        let mut counter_format = ExpressionFormat::default();

        if let Ok(env_debug_options) = std::env::var(RUSTC_COVERAGE_DEBUG_OPTIONS) {
            for setting_str in env_debug_options.replace(" ", "").replace("-", "_").split(",") {
                let mut setting = setting_str.splitn(2, "=");
                match setting.next() {
                    Some(option) if option == "counter_format" => {
                        if let Some(strval) = setting.next() {
                            counter_format = counter_format_option_val(strval);
                            debug!(
                                "{} env option `counter_format` is set to {:?}",
                                RUSTC_COVERAGE_DEBUG_OPTIONS, counter_format
                            );
                        } else {
                            bug!(
                                "`{}` option in environment variable {} requires one or more \
                                plus-separated choices (a non-empty subset of \
                                `id+block+operation`)",
                                option,
                                RUSTC_COVERAGE_DEBUG_OPTIONS
                            );
                        }
                    }
                    Some("") => {}
                    Some(invalid) => bug!(
                        "Unsupported setting `{}` in environment variable {}",
                        invalid,
                        RUSTC_COVERAGE_DEBUG_OPTIONS
                    ),
                    None => {}
                }
            }
        }

        Self { counter_format }
    }
}

fn counter_format_option_val(strval: &str) -> ExpressionFormat {
    let mut counter_format = ExpressionFormat { id: false, block: false, operation: false };
    let components = strval.splitn(3, "+");
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
pub(crate) struct DebugCounters {
    some_counters: Option<FxHashMap<ExpressionOperandId, DebugCounter>>,
}

impl DebugCounters {
    pub fn new() -> Self {
        Self { some_counters: None }
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
                CoverageKind::Expression { id, .. } => id.into(),
                _ => bug!(
                    "the given `CoverageKind` is not an counter or expression: {:?}",
                    counter_kind
                ),
            };
            counters
                .insert(id.into(), DebugCounter::new(counter_kind.clone(), some_block_label))
                .expect_none(
                    "attempt to add the same counter_kind to DebugCounters more than once",
                );
        }
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
                counters.get(&id.into())
            {
                return if counter_format.id {
                    format!("{}#{}", block_label, id.index())
                } else {
                    format!("{}", block_label)
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
                return format!("{}", self.format_counter_kind(counter_kind));
            }
        }
        format!("#{}", operand.index().to_string())
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
pub(crate) struct GraphvizData {
    some_bcb_to_coverage_spans_with_counters:
        Option<FxHashMap<BasicCoverageBlock, Vec<(CoverageSpan, CoverageKind)>>>,
    some_edge_to_counter: Option<FxHashMap<(BasicCoverageBlock, BasicBlock), CoverageKind>>,
}

impl GraphvizData {
    pub fn new() -> Self {
        Self { some_bcb_to_coverage_spans_with_counters: None, some_edge_to_counter: None }
    }

    pub fn enable(&mut self) {
        self.some_bcb_to_coverage_spans_with_counters = Some(FxHashMap::default());
        self.some_edge_to_counter = Some(FxHashMap::default());
    }

    pub fn is_enabled(&mut self) -> bool {
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
                .or_insert_with(|| Vec::new())
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
}

/// Generates the MIR pass `CoverageSpan`-specific spanview dump file.
pub(crate) fn dump_coverage_spanview(
    tcx: TyCtxt<'tcx>,
    mir_body: &mir::Body<'tcx>,
    basic_coverage_blocks: &CoverageGraph,
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

/// Converts the computed `BasicCoverageBlockData`s into `SpanViewable`s.
fn span_viewables(
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
pub(crate) fn dump_coverage_graphviz(
    tcx: TyCtxt<'tcx>,
    mir_body: &mir::Body<'tcx>,
    pass_name: &str,
    basic_coverage_blocks: &CoverageGraph,
    debug_counters: &DebugCounters,
    graphviz_data: &GraphvizData,
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
        )
    };
    let edge_labels = |from_bcb| {
        let from_bcb_data = &basic_coverage_blocks[from_bcb];
        let from_terminator = from_bcb_data.terminator(mir_body);
        from_terminator
            .kind
            .fmt_successor_labels()
            .iter()
            .map(|label| label.to_string())
            .collect::<Vec<_>>()
    };
    let graphviz_name = format!("Cov_{}_{}", def_id.krate.index(), def_id.index.index());
    let graphviz_writer =
        GraphvizWriter::new(basic_coverage_blocks, &graphviz_name, node_content, edge_labels);
    let mut file = pretty::create_dump_file(tcx, "dot", None, pass_name, &0, mir_source)
        .expect("Unexpected error creating BasicCoverageBlock graphviz DOT file");
    graphviz_writer
        .write_graphviz(tcx, &mut file)
        .expect("Unexpected error writing BasicCoverageBlock graphviz DOT file");
}

fn bcb_to_string_sections(
    tcx: TyCtxt<'tcx>,
    mir_body: &mir::Body<'tcx>,
    debug_counters: &DebugCounters,
    bcb_data: &BasicCoverageBlockData,
    some_coverage_spans_with_counters: Option<&Vec<(CoverageSpan, CoverageKind)>>,
) -> Vec<String> {
    let len = bcb_data.basic_blocks.len();
    let mut sections = Vec::new();
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

/// Returns a simple string representation of a `TerminatorKind` variant, indenpendent of any
/// values it might hold.
pub(crate) fn term_type(kind: &TerminatorKind<'tcx>) -> &'static str {
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
