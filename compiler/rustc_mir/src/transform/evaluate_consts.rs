//! Evaluates constants.

use crate::transform::MirPass;
use rustc_middle::mir;
use rustc_middle::mir::visit::MutVisitor;
use rustc_middle::ty::{self, TyCtxt};

pub struct EvaluateConsts;

impl<'tcx> MirPass<'tcx> for EvaluateConsts {
    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut mir::Body<'tcx>) {
        let def_id = body.source.def_id().expect_local();
        let param_env = tcx.param_env_reveal_all_normalized(def_id);
        ConstEval { tcx, param_env }.visit_body(body);
    }
}

struct ConstEval<'tcx> {
    tcx: TyCtxt<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
}

impl<'tcx> MutVisitor<'tcx> for ConstEval<'tcx> {
    #[inline]
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn visit_body(&mut self, body: &mut mir::Body<'tcx>) {
        for (bb, data) in body.basic_blocks_mut().iter_enumerated_mut() {
            self.visit_basic_block_data(bb, data);
        }
        body.required_consts.retain(|constant| {
            let ct = match constant.literal {
                mir::ConstantKind::Ty(ct) => ct,
                mir::ConstantKind::Val(..) => return false,
            };
            let unevaluated = match ct.val {
                ty::ConstKind::Unevaluated(unevaluated) => unevaluated,
                _ => return false,
            };
            self.tcx.const_eval_resolve(self.param_env, unevaluated, None).is_err()
        });
    }

    fn visit_constant(&mut self, constant: &mut mir::Constant<'tcx>, _: mir::Location) {
        let ct = match constant.literal {
            mir::ConstantKind::Ty(ct) => ct,
            mir::ConstantKind::Val(..) => return,
        };
        let unevaluated = match ct.val {
            ty::ConstKind::Unevaluated(unevaluated) => unevaluated,
            _ => return,
        };
        if let Ok(evaluated) = self.tcx.const_eval_resolve(self.param_env, unevaluated, None) {
            constant.literal = mir::ConstantKind::Val(evaluated, ct.ty);
        }
    }
}
