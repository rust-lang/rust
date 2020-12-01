use rustc_index::vec::IndexVec;
use rustc_infer::infer::{InferCtxt, NLLRegionVariableOrigin};
use rustc_middle::mir::visit::{MutVisitor, TyContext};
use rustc_middle::mir::{Body, Location, PlaceElem, Promoted};
use rustc_middle::ty::subst::SubstsRef;
use rustc_middle::ty::{self, Ty, TyCtxt, TypeFoldable};

/// Replaces all free regions appearing in the MIR with fresh
/// inference variables, returning the number of variables created.
pub fn renumber_mir<'tcx>(
    infcx: &InferCtxt<'_, 'tcx>,
    body: &mut Body<'tcx>,
    promoted: &mut IndexVec<Promoted, Body<'tcx>>,
) {
    debug!("renumber_mir()");
    debug!("renumber_mir: body.arg_count={:?}", body.arg_count);

    let mut visitor = NLLVisitor { infcx };

    for body in promoted.iter_mut() {
        visitor.visit_body(body);
    }

    visitor.visit_body(body);
}

/// Replaces all regions appearing in `value` with fresh inference
/// variables.
pub fn renumber_regions<'tcx, T>(infcx: &InferCtxt<'_, 'tcx>, value: T) -> T
where
    T: TypeFoldable<'tcx>,
{
    debug!("renumber_regions(value={:?})", value);

    infcx.tcx.fold_regions(value, &mut false, |_region, _depth| {
        let origin = NLLRegionVariableOrigin::Existential { from_forall: false };
        infcx.next_nll_region_var(origin)
    })
}

struct NLLVisitor<'a, 'tcx> {
    infcx: &'a InferCtxt<'a, 'tcx>,
}

impl<'a, 'tcx> NLLVisitor<'a, 'tcx> {
    fn renumber_regions<T>(&mut self, value: T) -> T
    where
        T: TypeFoldable<'tcx>,
    {
        renumber_regions(self.infcx, value)
    }
}

impl<'a, 'tcx> MutVisitor<'tcx> for NLLVisitor<'a, 'tcx> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.infcx.tcx
    }

    fn visit_ty(&mut self, ty: &mut Ty<'tcx>, ty_context: TyContext) {
        debug!("visit_ty(ty={:?}, ty_context={:?})", ty, ty_context);

        *ty = self.renumber_regions(ty);

        debug!("visit_ty: ty={:?}", ty);
    }

    fn process_projection_elem(
        &mut self,
        elem: PlaceElem<'tcx>,
        _: Location,
    ) -> Option<PlaceElem<'tcx>> {
        if let PlaceElem::Field(field, ty) = elem {
            let new_ty = self.renumber_regions(ty);

            if new_ty != ty {
                return Some(PlaceElem::Field(field, new_ty));
            }
        }

        None
    }

    fn visit_substs(&mut self, substs: &mut SubstsRef<'tcx>, location: Location) {
        debug!("visit_substs(substs={:?}, location={:?})", substs, location);

        *substs = self.renumber_regions(*substs);

        debug!("visit_substs: substs={:?}", substs);
    }

    fn visit_region(&mut self, region: &mut ty::Region<'tcx>, location: Location) {
        debug!("visit_region(region={:?}, location={:?})", region, location);

        let old_region = *region;
        *region = self.renumber_regions(&old_region);

        debug!("visit_region: region={:?}", region);
    }

    fn visit_const(&mut self, constant: &mut &'tcx ty::Const<'tcx>, _location: Location) {
        *constant = self.renumber_regions(&*constant);
    }
}
