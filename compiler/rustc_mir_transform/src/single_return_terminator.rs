use crate::MirPass;
use rustc_middle::mir::*;
use rustc_middle::ty::TyCtxt;

use super::simplify::simplify_cfg;

pub struct SingleReturnTerminator;

impl<'tcx> MirPass<'tcx> for SingleReturnTerminator {
    fn is_enabled(&self, sess: &rustc_session::Session) -> bool {
        sess.mir_opt_level() >= 1
    }

    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        let mut returns = 0;
        let mut return_terminator = None;
        let mut return_block = None;
        let mut has_extra_return_blocks = false;

        for (block, block_data) in body.basic_blocks.iter_enumerated() {
            let terminator = block_data.terminator();
            if let TerminatorKind::Return = terminator.kind {
                return_terminator = Some(terminator.clone());
                returns += 1;
                if block_data.statements.is_empty() {
                    if return_block.is_some() {
                        has_extra_return_blocks = true;
                    } else {
                        return_block = Some(block);
                    }
                }
            }
        }

        if returns < 2 {
            return;
        }
        assert!(return_terminator.is_some());

        let return_block = return_block.unwrap_or_else(|| {
            body.basic_blocks_mut().push(BasicBlockData::new(return_terminator))
        });

        for (bb, block) in body.basic_blocks_mut().iter_enumerated_mut() {
            if bb == return_block {
                continue;
            }
            let terminator = block.terminator_mut();
            if let TerminatorKind::Return = terminator.kind {
                terminator.kind = TerminatorKind::Goto { target: return_block };
            }
        }

        if has_extra_return_blocks {
            simplify_cfg(tcx, body);
        }
    }
}
