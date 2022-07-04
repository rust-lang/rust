use rustc_index::vec::IndexVec;
use rustc_infer::infer::{InferCtxt, NllRegionVariableOrigin};
use rustc_middle::mir::visit::{MutVisitor, TyContext};
use rustc_middle::mir::{Body, Location, Promoted};
use rustc_middle::mir::{Constant, ConstantKind};
use rustc_middle::ty::subst::SubstsRef;
use rustc_middle::ty::{self, Ty, TyCtxt, TypeFoldable};

/// Replaces all free regions appearing in the MIR with fresh
/// inference variables, returning the number of variables created.
#[instrument(skip(infcx, body, promoted), level = "debug")]
pub fn renumber_mir<'tcx>(
    infcx: &InferCtxt<'_, 'tcx>,
    body: &mut Body<'tcx>,
    promoted: &mut IndexVec<Promoted, Body<'tcx>>,
) {
    debug!(?body.arg_count);

    let mut visitor = NllVisitor { infcx };

    for body in promoted.iter_mut() {
        visitor.visit_body(body);
    }

    visitor.visit_body(body);
}

/// Replaces all regions appearing in `value` with fresh inference
/// variables.
#[instrument(skip(infcx), level = "debug")]
pub fn renumber_regions<'tcx, T>(infcx: &InferCtxt<'_, 'tcx>, value: T) -> T
where
    T: TypeFoldable<'tcx>,
{
    infcx.tcx.fold_regions(value, |_region, _depth| {
        let origin = NllRegionVariableOrigin::Existential { from_forall: false };
        infcx.next_nll_region_var(origin)
    })
}

struct NllVisitor<'a, 'tcx> {
    infcx: &'a InferCtxt<'a, 'tcx>,
}

impl<'a, 'tcx> NllVisitor<'a, 'tcx> {
    fn renumber_regions<T>(&mut self, value: T) -> T
    where
        T: TypeFoldable<'tcx>,
    {
        renumber_regions(self.infcx, value)
    }
}

impl<'a, 'tcx> MutVisitor<'tcx> for NllVisitor<'a, 'tcx> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.infcx.tcx
    }

    #[instrument(skip(self), level = "debug")]
    fn visit_ty(&mut self, ty: &mut Ty<'tcx>, ty_context: TyContext) {
        *ty = self.renumber_regions(*ty);

        debug!(?ty);
    }

    #[instrument(skip(self), level = "debug")]
    fn visit_substs(&mut self, substs: &mut SubstsRef<'tcx>, location: Location) {
        *substs = self.renumber_regions(*substs);

        debug!(?substs);
    }

    #[instrument(skip(self), level = "debug")]
    fn visit_region(&mut self, region: &mut ty::Region<'tcx>, location: Location) {
        let old_region = *region;
        *region = self.renumber_regions(old_region);

        debug!(?region);
    }

    #[instrument(skip(self), level = "debug")]
    fn visit_constant(&mut self, constant: &mut Constant<'tcx>, _location: Location) {
        let literal = constant.literal;
        debug!("{:#?}", literal);

        match literal {
            ConstantKind::Ty(ct) => {
                let ct = self.renumber_regions(ct);
                debug!("renumbered ct {:#?}", ct);

                constant.literal = ConstantKind::Ty(ct);
            }
            ConstantKind::Unevaluated(uv, ty) => {
                debug!("uv: {:#?}, ty: {:#?}", uv, ty);
                let uv = self.renumber_regions(uv);
                debug!("uv: {:#?}", uv);
                let ty = self.renumber_regions(ty);
                debug!("{:#?}", ty);
                constant.literal = ConstantKind::Unevaluated(uv, ty);
            }
            ConstantKind::Val(val, ty) => {
                let ty = self.renumber_regions(ty);
                constant.literal = ConstantKind::Val(val, ty);
            }
        }

        debug!("constant: {:#?}", constant);
    }
}
