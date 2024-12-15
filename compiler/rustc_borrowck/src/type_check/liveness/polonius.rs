use rustc_middle::mir::visit::{MutatingUseContext, PlaceContext, Visitor};
use rustc_middle::mir::{Body, Local, Location, Place};
use rustc_middle::ty::GenericArg;
use rustc_mir_dataflow::move_paths::{LookupResult, MoveData, MovePathIndex};
use tracing::debug;

use super::TypeChecker;
use crate::def_use::{self, DefUse};
use crate::location::{LocationIndex, LocationTable};

type VarPointRelation = Vec<(Local, LocationIndex)>;
type PathPointRelation = Vec<(MovePathIndex, LocationIndex)>;

/// Emit polonius facts for variable defs, uses, drops, and path accesses.
pub(super) fn emit_access_facts<'a, 'tcx>(
    typeck: &mut TypeChecker<'a, 'tcx>,
    body: &Body<'tcx>,
    move_data: &MoveData<'tcx>,
) {
    if let Some(facts) = typeck.all_facts.as_mut() {
        debug!("emit_access_facts()");

        let _prof_timer = typeck.infcx.tcx.prof.generic_activity("polonius_fact_generation");
        let location_table = typeck.location_table;

        let mut extractor = AccessFactsExtractor {
            var_defined_at: &mut facts.var_defined_at,
            var_used_at: &mut facts.var_used_at,
            var_dropped_at: &mut facts.var_dropped_at,
            path_accessed_at_base: &mut facts.path_accessed_at_base,
            location_table,
            move_data,
        };
        extractor.visit_body(body);

        for (local, local_decl) in body.local_decls.iter_enumerated() {
            debug!(
                "add use_of_var_derefs_origin facts - local={:?}, type={:?}",
                local, local_decl.ty
            );
            let universal_regions = &typeck.universal_regions;
            typeck.infcx.tcx.for_each_free_region(&local_decl.ty, |region| {
                let region_vid = universal_regions.to_region_vid(region);
                facts.use_of_var_derefs_origin.push((local, region_vid.into()));
            });
        }
    }
}

/// For every potentially drop()-touched region `region` in `local`'s type
/// (`kind`), emit a Polonius `use_of_var_derefs_origin(local, origin)` fact.
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

/// MIR visitor extracting point-wise facts about accesses.
struct AccessFactsExtractor<'a, 'tcx> {
    var_defined_at: &'a mut VarPointRelation,
    var_used_at: &'a mut VarPointRelation,
    location_table: &'a LocationTable,
    var_dropped_at: &'a mut VarPointRelation,
    move_data: &'a MoveData<'tcx>,
    path_accessed_at_base: &'a mut PathPointRelation,
}

impl<'tcx> AccessFactsExtractor<'_, 'tcx> {
    fn location_to_index(&self, location: Location) -> LocationIndex {
        self.location_table.mid_index(location)
    }
}

impl<'a, 'tcx> Visitor<'tcx> for AccessFactsExtractor<'a, 'tcx> {
    fn visit_local(&mut self, local: Local, context: PlaceContext, location: Location) {
        match def_use::categorize(context) {
            Some(DefUse::Def) => {
                debug!("AccessFactsExtractor - emit def");
                self.var_defined_at.push((local, self.location_to_index(location)));
            }
            Some(DefUse::Use) => {
                debug!("AccessFactsExtractor - emit use");
                self.var_used_at.push((local, self.location_to_index(location)));
            }
            Some(DefUse::Drop) => {
                debug!("AccessFactsExtractor - emit drop");
                self.var_dropped_at.push((local, self.location_to_index(location)));
            }
            _ => (),
        }
    }

    fn visit_place(&mut self, place: &Place<'tcx>, context: PlaceContext, location: Location) {
        self.super_place(place, context, location);

        match context {
            PlaceContext::NonMutatingUse(_)
            | PlaceContext::MutatingUse(MutatingUseContext::Borrow) => {
                let path = match self.move_data.rev_lookup.find(place.as_ref()) {
                    LookupResult::Exact(path) | LookupResult::Parent(Some(path)) => path,
                    _ => {
                        // There's no path access to emit.
                        return;
                    }
                };
                debug!("AccessFactsExtractor - emit path access ({path:?}, {location:?})");
                self.path_accessed_at_base.push((path, self.location_to_index(location)));
            }

            _ => {}
        }
    }
}
