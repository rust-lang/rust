use rustc_middle::mir::Local;
use rustc_middle::ty::GenericArg;
use tracing::debug;

use crate::type_check::TypeChecker;

/// For every potentially drop()-touched region `region` in `local`'s type
/// (`kind`), emit a Polonius `drop_of_var_derefs_origin(local, origin)` fact.
pub(super) fn emit_drop_facts<'tcx>(
    typeck: &mut TypeChecker<'_, 'tcx>,
    local: Local,
    kind: &GenericArg<'tcx>,
) {
    debug!("emit_drop_facts(local={:?}, kind={:?}", local, kind);
    if let Some(facts) = typeck.all_facts.as_mut() {
        let _prof_timer = typeck.infcx.tcx.prof.generic_activity("polonius_fact_generation");
        let universal_regions = &typeck.universal_regions;
        typeck.infcx.tcx.for_each_free_region(kind, |drop_live_region| {
            let region_vid = universal_regions.to_region_vid(drop_live_region);
            facts.drop_of_var_derefs_origin.push((local, region_vid.into()));
        });
    }
}
