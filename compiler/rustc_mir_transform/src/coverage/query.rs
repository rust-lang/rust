use rustc_middle::mir::coverage::*;
use rustc_middle::mir::{self, CoverageInfo};
use rustc_middle::query::Providers;
use rustc_middle::ty::{self, TyCtxt};
use rustc_span::def_id::DefId;

/// A `query` provider for retrieving coverage information injected into MIR.
pub(crate) fn provide(providers: &mut Providers) {
    providers.coverageinfo = |tcx, def_id| coverageinfo(tcx, def_id);
    providers.covered_code_regions = |tcx, def_id| covered_code_regions(tcx, def_id);
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
/// are still included in the total `num_counters` or `num_expressions`.)
fn coverageinfo<'tcx>(tcx: TyCtxt<'tcx>, instance_def: ty::InstanceDef<'tcx>) -> CoverageInfo {
    let mir_body = tcx.instance_mir(instance_def);

    if let Some(info) = &mir_body.coverage_info {
        CoverageInfo {
            num_counters: info.num_counters,
            num_expressions: info.expressions.len() as u32,
        }
    } else {
        CoverageInfo { num_counters: 0, num_expressions: 0 }
    }
}

fn covered_code_regions(tcx: TyCtxt<'_>, def_id: DefId) -> Vec<&CodeRegion> {
    let body = mir_body(tcx, def_id);
    let Some(info) = body.coverage_info.as_ref() else { return vec![] };
    info.regions.iter().map(|region| &region.code_region).collect()
}

/// This function ensures we obtain the correct MIR for the given item irrespective of
/// whether that means const mir or runtime mir. For `const fn` this opts for runtime
/// mir.
fn mir_body(tcx: TyCtxt<'_>, def_id: DefId) -> &mir::Body<'_> {
    let def = ty::InstanceDef::Item(def_id);
    tcx.instance_mir(def)
}
