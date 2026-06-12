//! Optimize `Clone` calls created from `use` statements into direct copies for codegen MIR.

use rustc_middle::mir::*;
use rustc_middle::ty::{self, TyCtxt};
use rustc_session::Session;

use crate::deref_separator::deref_finder;

pub(super) struct OptimizeUseClone;

impl<'tcx> crate::MirPass<'tcx> for OptimizeUseClone {
    fn is_enabled(&self, sess: &Session) -> bool {
        sess.mir_opt_level() >= 4
    }

    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        if !tcx.features().ergonomic_clones() {
            return;
        }

        let typing_env = body.typing_env(tcx);
        let basic_blocks = body.basic_blocks.as_mut();
        for block in basic_blocks {
            let TerminatorKind::Call {
                args: [arg],
                destination,
                target: Some(target),
                call_source: CallSource::Use,
                ..
            } = &block.terminator().kind
            else {
                continue;
            };

            let arg_ty = arg.node.ty(&body.local_decls, tcx);
            let ty::Ref(_, inner_ty, Mutability::Not) = *arg_ty.kind() else {
                continue;
            };

            if !tcx.type_is_copy_modulo_regions(typing_env, inner_ty) {
                continue;
            }

            let Some(arg_place) = arg.node.place() else {
                continue;
            };

            let target = *target;
            block.statements.push(Statement::new(
                block.terminator().source_info,
                StatementKind::Assign(Box::new((
                    *destination,
                    Rvalue::Use(Operand::Copy(tcx.mk_place_deref(arg_place)), WithRetag::Yes),
                ))),
            ));
            block.terminator_mut().kind = TerminatorKind::Goto { target };
        }

        deref_finder(tcx, body, false);
    }

    fn is_required(&self) -> bool {
        false
    }
}
