//! Normalizes MIR in RevealAll mode.

use crate::MirPass;
use rustc_middle::mir::visit::*;
use rustc_middle::mir::*;
use rustc_middle::ty::{self, Ty, TyCtxt};

pub struct RevealAll;

impl<'tcx> MirPass<'tcx> for RevealAll {
    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        let param_env = tcx.param_env_reveal_all_normalized(body.source.def_id());
        RevealAllVisitor { tcx, param_env }.visit_body_preserves_cfg(body);
    }
}

struct RevealAllVisitor<'tcx> {
    tcx: TyCtxt<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
}

impl<'tcx> MutVisitor<'tcx> for RevealAllVisitor<'tcx> {
    #[inline]
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    #[inline]
    fn visit_place(
        &mut self,
        place: &mut Place<'tcx>,
        _context: PlaceContext,
        _location: Location,
    ) {
        // Performance optimization: don't reintern if there is no `OpaqueCast` to remove.
        if place.projection.iter().all(|elem| !matches!(elem, ProjectionElem::OpaqueCast(_))) {
            return;
        }
        // `OpaqueCast` projections are only needed if there are opaque types on which projections are performed.
        // After the `RevealAll` pass, all opaque types are replaced with their hidden types, so we don't need these
        // projections anymore.
        place.projection = self.tcx.mk_place_elems(
            &place
                .projection
                .into_iter()
                .filter(|elem| !matches!(elem, ProjectionElem::OpaqueCast(_)))
                .collect::<Vec<_>>(),
        );
        self.super_place(place, _context, _location);
    }

    #[inline]
    fn visit_constant(&mut self, constant: &mut ConstOperand<'tcx>, location: Location) {
        // We have to use `try_normalize_erasing_regions` here, since it's
        // possible that we visit impossible-to-satisfy where clauses here,
        // see #91745
        if let Ok(c) = self.tcx.try_normalize_erasing_regions(self.param_env, constant.const_) {
            constant.const_ = c;
        }
        self.super_constant(constant, location);
    }

    #[inline]
    fn visit_ty(&mut self, ty: &mut Ty<'tcx>, _: TyContext) {
        // We have to use `try_normalize_erasing_regions` here, since it's
        // possible that we visit impossible-to-satisfy where clauses here,
        // see #91745
        if let Ok(t) = self.tcx.try_normalize_erasing_regions(self.param_env, *ty) {
            *ty = t;
        }
    }
}
