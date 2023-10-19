use super::*;

use rustc_data_structures::captures::Captures;
use rustc_middle::mir::coverage::*;
use rustc_middle::mir::{Body, Coverage, CoverageIdsInfo};
use rustc_middle::query::Providers;
use rustc_middle::ty::{self, TyCtxt};

/// A `query` provider for retrieving coverage information injected into MIR.
pub(crate) fn provide(providers: &mut Providers) {
    providers.coverage_ids_info = |tcx, def_id| coverage_ids_info(tcx, def_id);
}

/// Query implementation for `coverage_ids_info`.
fn coverage_ids_info<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance_def: ty::InstanceDef<'tcx>,
) -> CoverageIdsInfo {
    let mir_body = tcx.instance_mir(instance_def);

    let max_counter_id = all_coverage_in_mir_body(mir_body)
        .filter_map(|coverage| match coverage.kind {
            CoverageKind::CounterIncrement { id } => Some(id),
            _ => None,
        })
        .max()
        .unwrap_or(CounterId::START);

    CoverageIdsInfo { max_counter_id }
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
