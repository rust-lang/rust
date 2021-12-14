//! Normalizes MIR in RevealAll mode.

use crate::MirPass;
use rustc_middle::mir::visit::*;
use rustc_middle::mir::*;
use rustc_middle::ty::{self, Ty, TyCtxt};

pub struct RevealAll;

impl<'tcx> MirPass<'tcx> for RevealAll {
    fn is_enabled(&self, sess: &rustc_session::Session) -> bool {
        sess.opts.mir_opt_level() >= 3 || super::inline::Inline.is_enabled(sess)
    }

    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        // Do not apply this transformation to generators.
        if body.generator.is_some() {
            return;
        }

        let param_env = tcx.param_env_reveal_all_normalized(body.source.def_id());
        RevealAllVisitor { tcx, param_env }.visit_body(body);
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
        // We have to use `try_normalize_erasing_regions` here, since it's
        // possible that we visit impossible-to-satisfy where clauses here,
        // see #91745
        *ty = self.tcx.try_normalize_erasing_regions(self.param_env, *ty).unwrap_or(ty);
    }
}
