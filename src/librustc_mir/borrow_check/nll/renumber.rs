use rustc::ty::subst::SubstsRef;
use rustc::ty::{self, ClosureSubsts, GeneratorSubsts, Ty, TypeFoldable};
use rustc::mir::{Location, Body};
use rustc::mir::visit::{MutVisitor, TyContext};
use rustc::infer::{InferCtxt, NLLRegionVariableOrigin};

/// Replaces all free regions appearing in the MIR with fresh
/// inference variables, returning the number of variables created.
pub fn renumber_mir<'tcx>(infcx: &InferCtxt<'_, 'tcx>, body: &mut Body<'tcx>) {
    debug!("renumber_mir()");
    debug!("renumber_mir: body.arg_count={:?}", body.arg_count);

    let mut visitor = NLLVisitor { infcx };
    visitor.visit_body(body);
}

/// Replaces all regions appearing in `value` with fresh inference
/// variables.
pub fn renumber_regions<'tcx, T>(infcx: &InferCtxt<'_, 'tcx>, value: &T) -> T
where
    T: TypeFoldable<'tcx>,
{
    debug!("renumber_regions(value={:?})", value);

    infcx
        .tcx
        .fold_regions(value, &mut false, |_region, _depth| {
            let origin = NLLRegionVariableOrigin::Existential;
            infcx.next_nll_region_var(origin)
        })
}

struct NLLVisitor<'a, 'tcx> {
    infcx: &'a InferCtxt<'a, 'tcx>,
}

impl<'a, 'tcx> NLLVisitor<'a, 'tcx> {
    fn renumber_regions<T>(&mut self, value: &T) -> T
    where
        T: TypeFoldable<'tcx>,
    {
        renumber_regions(self.infcx, value)
    }
}

impl<'a, 'tcx> MutVisitor<'tcx> for NLLVisitor<'a, 'tcx> {
    fn visit_body(&mut self, body: &mut Body<'tcx>) {
        for promoted in body.promoted.iter_mut() {
            self.visit_body(promoted);
        }

        self.super_body(body);
    }

    fn visit_ty(&mut self, ty: &mut Ty<'tcx>, ty_context: TyContext) {
        debug!("visit_ty(ty={:?}, ty_context={:?})", ty, ty_context);

        *ty = self.renumber_regions(ty);

        debug!("visit_ty: ty={:?}", ty);
    }

    fn visit_substs(&mut self, substs: &mut SubstsRef<'tcx>, location: Location) {
        debug!("visit_substs(substs={:?}, location={:?})", substs, location);

        *substs = self.renumber_regions(&{ *substs });

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

    fn visit_generator_substs(&mut self,
                              substs: &mut GeneratorSubsts<'tcx>,
                              location: Location) {
        debug!(
            "visit_generator_substs(substs={:?}, location={:?})",
            substs,
            location,
        );

        *substs = self.renumber_regions(substs);

        debug!("visit_generator_substs: substs={:?}", substs);
    }

    fn visit_closure_substs(&mut self, substs: &mut ClosureSubsts<'tcx>, location: Location) {
        debug!(
            "visit_closure_substs(substs={:?}, location={:?})",
            substs,
            location
        );

        *substs = self.renumber_regions(substs);

        debug!("visit_closure_substs: substs={:?}", substs);
    }
}
