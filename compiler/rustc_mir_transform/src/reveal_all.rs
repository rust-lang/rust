//! Normalizes MIR in RevealAll mode.

use crate::MirPass;
use rustc_middle::mir::visit::*;
use rustc_middle::mir::*;
use rustc_middle::ty::{self, Ty, TyCtxt};

pub struct RevealAll;

impl<'tcx> MirPass<'tcx> for RevealAll {
    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        // This pass must run before inlining, since we insert callee bodies in RevealAll mode.
        // Do not apply this transformation to generators.
        if (tcx.sess.mir_opt_level() >= 3 || super::inline::is_enabled(tcx))
            && body.generator.is_none()
        {
            let param_env = tcx.param_env_reveal_all_normalized(body.source.def_id());
            RevealAllVisitor { tcx, param_env }.visit_body(body);
        }
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
    fn visit_ty(&mut self, ty: &mut Ty<'tcx>, _: TyContext) {
        *ty = self.tcx.normalize_erasing_regions(self.param_env, ty);
    }

    #[inline]
    fn process_projection_elem(
        &mut self,
        elem: PlaceElem<'tcx>,
        _: Location,
    ) -> Option<PlaceElem<'tcx>> {
        match elem {
            PlaceElem::Field(field, ty) => {
                let new_ty = self.tcx.normalize_erasing_regions(self.param_env, ty);
                if ty != new_ty { Some(PlaceElem::Field(field, new_ty)) } else { None }
            }
            // None of those contain a Ty.
            PlaceElem::Index(..)
            | PlaceElem::Deref
            | PlaceElem::ConstantIndex { .. }
            | PlaceElem::Subslice { .. }
            | PlaceElem::Downcast(..) => None,
        }
    }
}
