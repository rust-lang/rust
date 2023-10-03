use super::*;

use rustc_data_structures::captures::Captures;
use rustc_middle::mir::coverage::*;
use rustc_middle::mir::{self, Body, Coverage, CoverageInfo};
use rustc_middle::query::Providers;
use rustc_middle::ty::{self, TyCtxt};
use rustc_span::def_id::DefId;

/// A `query` provider for retrieving coverage information injected into MIR.
pub(crate) fn provide(providers: &mut Providers) {
    providers.coverageinfo = |tcx, def_id| coverageinfo(tcx, def_id);
    providers.covered_code_regions = |tcx, def_id| covered_code_regions(tcx, def_id);
}

/// Coverage codegen needs to know the total number of counter IDs and expression IDs that have
/// been used by a function's coverage mappings. These totals are used to create vectors to hold
/// the relevant counter and expression data, and the maximum counter ID (+ 1) is also needed by
/// the `llvm.instrprof.increment` intrinsic.
///
/// MIR optimization may split and duplicate some BasicBlock sequences, or optimize out some code
/// including injected counters. (It is OK if some counters are optimized out, but those counters
/// are still included in the total `num_counters` or `num_expressions`.) Simply counting the
/// calls may not work; but computing the number of counters or expressions by adding `1` to the
/// highest ID (for a given instrumented function) is valid.
///
/// It's possible for a coverage expression to remain in MIR while one or both of its operands
/// have been optimized away. To avoid problems in codegen, we include those operands' IDs when
/// determining the maximum counter/expression ID, even if the underlying counter/expression is
/// no longer present.
struct CoverageVisitor {
    max_counter_id: CounterId,
    max_expression_id: ExpressionId,
}

impl CoverageVisitor {
    /// Updates `max_counter_id` to the maximum encountered counter ID.
    #[inline(always)]
    fn update_max_counter_id(&mut self, counter_id: CounterId) {
        self.max_counter_id = self.max_counter_id.max(counter_id);
    }

    /// Updates `max_expression_id` to the maximum encountered expression ID.
    #[inline(always)]
    fn update_max_expression_id(&mut self, expression_id: ExpressionId) {
        self.max_expression_id = self.max_expression_id.max(expression_id);
    }

    fn update_from_expression_operand(&mut self, operand: Operand) {
        match operand {
            Operand::Counter(id) => self.update_max_counter_id(id),
            Operand::Expression(id) => self.update_max_expression_id(id),
            Operand::Zero => {}
        }
    }

    fn visit_body(&mut self, body: &Body<'_>) {
        for coverage in all_coverage_in_mir_body(body) {
            self.visit_coverage(coverage);
        }
    }

    fn visit_coverage(&mut self, coverage: &Coverage) {
        match coverage.kind {
            CoverageKind::Counter { id, .. } => self.update_max_counter_id(id),
            CoverageKind::Expression { id, lhs, rhs, .. } => {
                self.update_max_expression_id(id);
                self.update_from_expression_operand(lhs);
                self.update_from_expression_operand(rhs);
            }
            CoverageKind::Unreachable => {}
        }
    }
}

fn coverageinfo<'tcx>(tcx: TyCtxt<'tcx>, instance_def: ty::InstanceDef<'tcx>) -> CoverageInfo {
    let mir_body = tcx.instance_mir(instance_def);

    let mut coverage_visitor = CoverageVisitor {
        max_counter_id: CounterId::START,
        max_expression_id: ExpressionId::START,
    };

    coverage_visitor.visit_body(mir_body);

    // Add 1 to the highest IDs to get the total number of IDs.
    CoverageInfo {
        num_counters: (coverage_visitor.max_counter_id + 1).as_u32(),
        num_expressions: (coverage_visitor.max_expression_id + 1).as_u32(),
    }
}

fn covered_code_regions(tcx: TyCtxt<'_>, def_id: DefId) -> Vec<&CodeRegion> {
    let body = mir_body(tcx, def_id);
    all_coverage_in_mir_body(body)
        // Coverage statements have a list of code regions (possibly empty).
        .flat_map(|coverage| coverage.code_regions.as_slice())
        .collect()
}

fn all_coverage_in_mir_body<'a, 'tcx>(
    body: &'a Body<'tcx>,
) -> impl Iterator<Item = &'a Coverage> + Captures<'tcx> {
    body.basic_blocks.iter().flat_map(|bb_data| &bb_data.statements).filter_map(|statement| {
        match statement.kind {
            StatementKind::Coverage(box ref coverage) if !is_inlined(body, statement) => {
                Some(coverage)
            }
            _ => None,
        }
    })
}

fn is_inlined(body: &Body<'_>, statement: &Statement<'_>) -> bool {
    let scope_data = &body.source_scopes[statement.source_info.scope];
    scope_data.inlined.is_some() || scope_data.inlined_parent_scope.is_some()
}

/// This function ensures we obtain the correct MIR for the given item irrespective of
/// whether that means const mir or runtime mir. For `const fn` this opts for runtime
/// mir.
fn mir_body(tcx: TyCtxt<'_>, def_id: DefId) -> &mir::Body<'_> {
    let def = ty::InstanceDef::Item(def_id);
    tcx.instance_mir(def)
}
