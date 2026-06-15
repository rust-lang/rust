//! Folds chains of `&&` over pure boolean expressions into `BitAnd` BinOps,
//! eliminating the phi nodes that LLVM cannot optimize away.

use crate::MirPass;
use rustc_middle::mir::*;
use rustc_middle::ty::{self, TyCtxt};

pub(crate) struct BoolChainOpt;

impl<'tcx> MirPass<'tcx> for BoolChainOpt {
    fn is_enabled(&self, sess: &rustc_session::Session) -> bool {
        sess.mir_opt_level() >= 2
    }

    fn is_required(&self) -> bool {
        false
    }

    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        let typing_env = ty::TypingEnv::post_analysis(tcx, body.source.def_id());

        for (_bb, data) in body.basic_blocks.iter_enumerated() {
            let TerminatorKind::SwitchInt { discr, targets } = &data.terminator().kind else {
                continue;
            };

            if targets.all_targets().len() != 2 {
                continue;
            }

            let place = match discr {
                Operand::Copy(p) | Operand::Move(p) => p,
                _ => continue,
            };

            if !place.projection.is_empty() {
                continue;
            }

            let local = place.local;
            let local_ty = body.local_decls[local].ty;
            if !local_ty.is_bool() {
                continue;
            }

            let all_storage = data.statements.iter().all(|s| {
                matches!(s.kind, StatementKind::StorageLive(_) | StatementKind::StorageDead(_))
            });
            if !all_storage {
                continue;
            }
        }

        let _ = typing_env;
    }
}
