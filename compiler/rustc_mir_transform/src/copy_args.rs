use rustc_middle::mir::*;
use rustc_middle::ty::TyCtxt;
use rustc_session::config::OptLevel;

pub(super) struct CopyArgs;

impl<'tcx> crate::MirPass<'tcx> for CopyArgs {
    fn is_enabled(&self, sess: &rustc_session::Session) -> bool {
        sess.opts.optimize != OptLevel::No && sess.opts.incremental.is_none()
    }

    fn run_pass(&self, _: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        for (_, block) in body.basic_blocks.as_mut_preserves_cfg().iter_enumerated_mut() {
            if let TerminatorKind::Call { ref mut args, .. } = block.terminator_mut().kind {
                for arg in args {
                    if let Operand::Move(place) = arg.node {
                        if place.is_indirect() {
                            continue;
                        }
                        let Some(local) = place.as_local() else {
                            continue
                        };
                        if 1 <= local.index() && local.index() <= body.arg_count {
                            arg.node = Operand::Copy(place);
                        }
                    }
                }
            };
        }
    }

    fn is_required(&self) -> bool {
        false
    }
}
