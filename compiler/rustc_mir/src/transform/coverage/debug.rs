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

pub const NESTED_INDENT: &str = "    ";

const RUSTC_COVERAGE_DEBUG_OPTIONS: &str = "RUSTC_COVERAGE_DEBUG_OPTIONS";

pub(crate) fn debug_options<'a>() -> &'a DebugOptions {
    static DEBUG_OPTIONS: SyncOnceCell<DebugOptions> = SyncOnceCell::new();

    &DEBUG_OPTIONS.get_or_init(|| DebugOptions::new())
}

#[derive(Debug, Clone)]
pub(crate) struct DebugOptions {
    pub allow_unused_expressions: bool,
    counter_format: ExpressionFormat,
}

impl DebugOptions {
    fn new() -> Self {
        let mut allow_unused_expressions = true;
        let mut counter_format = ExpressionFormat::default();

        if let Ok(env_debug_options) = std::env::var(RUSTC_COVERAGE_DEBUG_OPTIONS) {
            for setting_str in env_debug_options.replace(" ", "").replace("-", "_").split(",") {
                let mut setting = setting_str.splitn(2, "=");
                match setting.next() {
                    Some(option) if option == "allow_unused_expressions" => {
                        allow_unused_expressions = bool_option_val(option, setting.next());
                        debug!(
                            "{} env option `allow_unused_expressions` is set to {}",
                            RUSTC_COVERAGE_DEBUG_OPTIONS, allow_unused_expressions
                        );
                    }
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

pub(crate) struct GraphvizData {
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
        self.some_bcb_to_coverage_spans_with_counters = Some(FxHashMap::default());
        self.some_bcb_to_dependency_counters = Some(FxHashMap::default());
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

    pub fn add_bcb_dependency_counter(
        &mut self,
        bcb: BasicCoverageBlock,
        counter_kind: &CoverageKind,
    ) {
        if let Some(bcb_to_dependency_counters) = self.some_bcb_to_dependency_counters.as_mut() {
            bcb_to_dependency_counters
                .entry(bcb)
                .or_insert_with(|| Vec::new())
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
            edge_to_counter.insert((from_bcb, to_bb), counter_kind.clone()).expect_none(
                "invalid attempt to insert more than one edge counter for the same edge",
            );
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

pub(crate) struct UsedExpressions {
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
        self.some_used_expression_operands = Some(FxHashMap::default());
        self.some_unused_expressions = Some(Vec::new());
    }

    pub fn is_enabled(&mut self) -> bool {
        self.some_used_expression_operands.is_some()
    }

    pub fn add_expression_operands(&mut self, expression: &CoverageKind) {
        if let Some(used_expression_operands) = self.some_used_expression_operands.as_mut() {
            if let CoverageKind::Expression { id, lhs, rhs, .. } = *expression {
                used_expression_operands.entry(lhs).or_insert_with(|| Vec::new()).push(id);
                used_expression_operands.entry(rhs).or_insert_with(|| Vec::new()).push(id);
            }
        }
    }

    pub fn expression_is_used(&mut self, expression: &CoverageKind) -> bool {
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

pub(crate) fn dump_coverage_graphviz(
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
        edge_labels.retain(|label| label.to_string() != "unreachable");
        let edge_counters = from_terminator
            .successors()
            .map(|&successor_bb| graphviz_data.get_edge_counter(from_bcb, successor_bb));
        edge_labels
            .iter()
            .zip(edge_counters)
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
