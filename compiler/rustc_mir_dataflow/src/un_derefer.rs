use rustc_data_structures::fx::FxHashMap;
use rustc_middle::mir::*;
use rustc_middle::ty::TyCtxt;

/// Used for reverting changes made by `DerefSeparator`
pub struct UnDerefer<'tcx> {
    pub tcx: TyCtxt<'tcx>,
    pub derefer_sidetable: FxHashMap<Local, Place<'tcx>>,
}

impl<'tcx> UnDerefer<'tcx> {
    #[inline]
    pub fn derefer(&self, place: PlaceRef<'tcx>, body: &Body<'tcx>) -> Option<Place<'tcx>> {
        let reffed = self.derefer_sidetable.get(&place.local)?;

        let new_place = reffed.project_deeper(place.projection, self.tcx);
        if body.local_decls[new_place.local].is_deref_temp() {
            return self.derefer(new_place.as_ref(), body);
        }
        Some(new_place)
    }
}
