use super::graph::BasicCoverageBlock;
use super::spans::CoverageSpan;

use rustc_data_structures::fx::FxHashMap;
use rustc_middle::mir::coverage::*;
use rustc_middle::mir::{BasicBlock, TerminatorKind};

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
    pub simplify_expressions: bool,
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
                        debug!(
                            "{} env option `allow_unused_expressions` is set to {}",
                            RUSTC_COVERAGE_DEBUG_OPTIONS, allow_unused_expressions
                        );
                    }
                    Some(option) if option == "simplify_expressions" => {
                        simplify_expressions = bool_option_val(option, setting.next());
                        debug!(
                            "{} env option `simplify_expressions` is set to {}",
                            RUSTC_COVERAGE_DEBUG_OPTIONS, simplify_expressions
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

        Self { allow_unused_expressions, simplify_expressions, counter_format }
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

    pub fn validate_expression_is_used(
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

    pub fn check_no_unused(&self, debug_counters: &DebugCounters) {
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

                if debug_options().simplify_expressions || debug_options().allow_unused_expressions
                {
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
    }
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
