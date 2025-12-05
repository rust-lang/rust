//! Normalizes MIR in `TypingMode::PostAnalysis` mode, most notably revealing
//! its opaques. We also only normalize specializable associated items once in
//! `PostAnalysis` mode.

use rustc_middle::mir::visit::*;
use rustc_middle::mir::*;
use rustc_middle::ty::{self, Ty, TyCtxt};

pub(super) struct PostAnalysisNormalize;

impl<'tcx> crate::MirPass<'tcx> for PostAnalysisNormalize {
    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        // FIXME(#132279): This is used during the phase transition from analysis
        // to runtime, so we have to manually specify the correct typing mode.
        let typing_env = ty::TypingEnv::post_analysis(tcx, body.source.def_id());
        PostAnalysisNormalizeVisitor { tcx, typing_env }.visit_body_preserves_cfg(body);
    }

    fn is_required(&self) -> bool {
        true
    }
}

struct PostAnalysisNormalizeVisitor<'tcx> {
    tcx: TyCtxt<'tcx>,
    typing_env: ty::TypingEnv<'tcx>,
}

impl<'tcx> MutVisitor<'tcx> for PostAnalysisNormalizeVisitor<'tcx> {
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
        if !self.tcx.next_trait_solver_globally() {
            // `OpaqueCast` projections are only needed if there are opaque types on which projections
            // are performed. After the `PostAnalysisNormalize` pass, all opaque types are replaced with their
            // hidden types, so we don't need these projections anymore.
            //
            // Performance optimization: don't reintern if there is no `OpaqueCast` to remove.
            if place.projection.iter().any(|elem| matches!(elem, ProjectionElem::OpaqueCast(_))) {
                place.projection = self.tcx.mk_place_elems(
                    &place
                        .projection
                        .into_iter()
                        .filter(|elem| !matches!(elem, ProjectionElem::OpaqueCast(_)))
                        .collect::<Vec<_>>(),
                );
            };
        }
        self.super_place(place, _context, _location);
    }

    #[inline]
    fn visit_const_operand(&mut self, constant: &mut ConstOperand<'tcx>, location: Location) {
        // We have to use `try_normalize_erasing_regions` here, since it's
        // possible that we visit impossible-to-satisfy where clauses here,
        // see #91745
        if let Ok(c) = self.tcx.try_normalize_erasing_regions(self.typing_env, constant.const_) {
            constant.const_ = c;
        }
        self.super_const_operand(constant, location);
    }

    #[inline]
    fn visit_ty(&mut self, ty: &mut Ty<'tcx>, _: TyContext) {
        // We have to use `try_normalize_erasing_regions` here, since it's
        // possible that we visit impossible-to-satisfy where clauses here,
        // see #91745
        if let Ok(t) = self.tcx.try_normalize_erasing_regions(self.typing_env, *ty) {
            *ty = t;
        }
    }
}
