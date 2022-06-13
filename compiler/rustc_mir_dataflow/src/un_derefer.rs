use rustc_data_structures::stable_map::FxHashMap;
use rustc_middle::mir::*;
use rustc_middle::ty::TyCtxt;

/// Used for reverting changes made by `DerefSeparator`
pub struct UnDerefer<'tcx> {
    pub tcx: TyCtxt<'tcx>,
    pub derefer_sidetable: FxHashMap<Local, Place<'tcx>>,
}

impl<'tcx> UnDerefer<'tcx> {
    pub fn derefer(&self, place: PlaceRef<'tcx>) -> Option<Place<'tcx>> {
        let reffed = self.derefer_sidetable.get(&place.local)?;

        let new_place = reffed.project_deeper(place.projection, self.tcx);
        Some(new_place)
    }

    pub fn ref_finder(&mut self, body: &Body<'tcx>) {
        for (_bb, data) in body.basic_blocks().iter_enumerated() {
            for stmt in data.statements.iter() {
                match stmt.kind {
                    StatementKind::Assign(box (place, Rvalue::VirtualRef(reffed))) => {
                        self.derefer_sidetable.insert(place.local, reffed);
                    }
                    _ => (),
                }
            }
        }
    }
}
