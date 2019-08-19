use crate::borrow_check::location::{LocationIndex, LocationTable};
use crate::util::liveness::{categorize, DefUse};
use rustc::mir::visit::{PlaceContext, Visitor};
use rustc::mir::{Body, Local, Location};
use rustc::ty::subst::Kind;
use rustc::ty::Ty;

use super::TypeChecker;

type VarPointRelations = Vec<(Local, LocationIndex)>;

struct LivenessPointFactsExtractor<'me> {
    var_defined: &'me mut VarPointRelations,
    var_used: &'me mut VarPointRelations,
    location_table: &'me LocationTable,
}

// A Visitor to walk through the MIR and extract point-wise facts
impl LivenessPointFactsExtractor<'_> {
    fn location_to_index(&self, location: Location) -> LocationIndex {
        self.location_table.mid_index(location)
    }

    fn insert_def(&mut self, local: Local, location: Location) {
        debug!("LivenessFactsExtractor::insert_def()");
        self.var_defined.push((local, self.location_to_index(location)));
    }

    fn insert_use(&mut self, local: Local, location: Location) {
        debug!("LivenessFactsExtractor::insert_use()");
        self.var_used.push((local, self.location_to_index(location)));
    }
}

impl Visitor<'tcx> for LivenessPointFactsExtractor<'_> {
    fn visit_local(&mut self, &local: &Local, context: PlaceContext, location: Location) {
        match categorize(context) {
            Some(DefUse::Def) => self.insert_def(local, location),
            Some(DefUse::Use) => self.insert_use(local, location),
            _ => (),
            // NOTE: Drop handling is now done in trace()
        }
    }
}

fn add_var_uses_regions(typeck: &mut TypeChecker<'_, 'tcx>, local: Local, ty: Ty<'tcx>) {
    debug!("add_regions(local={:?}, type={:?})", local, ty);
    typeck.tcx().for_each_free_region(&ty, |region| {
        let region_vid = typeck.borrowck_context.universal_regions.to_region_vid(region);
        debug!("add_regions for region {:?}", region_vid);
        if let Some(facts) = typeck.borrowck_context.all_facts {
            facts.var_uses_region.push((local, region_vid));
        }
    });
}

pub(super) fn populate_var_liveness_facts(
    typeck: &mut TypeChecker<'_, 'tcx>,
    mir: &Body<'tcx>,
    location_table: &LocationTable,
) {
    debug!("populate_var_liveness_facts()");

    if let Some(facts) = typeck.borrowck_context.all_facts.as_mut() {
        LivenessPointFactsExtractor {
            var_defined: &mut facts.var_defined,
            var_used: &mut facts.var_used,
            location_table,
        }
        .visit_body(mir);
    }

    for (local, local_decl) in mir.local_decls.iter_enumerated() {
        add_var_uses_regions(typeck, local, local_decl.ty);
    }
}

// For every potentially drop()-touched region `region` in `local`'s type
// (`kind`), emit a Polonius `var_drops_region(local, region)` fact.
pub(super) fn add_var_drops_regions(
    typeck: &mut TypeChecker<'_, 'tcx>,
    local: Local,
    kind: &Kind<'tcx>,
) {
    debug!("add_var_drops_region(local={:?}, kind={:?}", local, kind);
    let tcx = typeck.tcx();

    tcx.for_each_free_region(kind, |drop_live_region| {
        let region_vid = typeck.borrowck_context.universal_regions.to_region_vid(drop_live_region);
        if let Some(facts) = typeck.borrowck_context.all_facts.as_mut() {
            facts.var_drops_region.push((local, region_vid));
        };
    });
}
