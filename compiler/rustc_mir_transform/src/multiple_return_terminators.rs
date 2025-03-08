//! This pass removes jumps to basic blocks containing only a return, and replaces them with a
//! return instead.

use rustc_middle::mir::*;
use rustc_middle::ty::TyCtxt;
use smallvec::SmallVec;

pub(super) struct MultipleReturnTerminators;

impl<'tcx> crate::MirPass<'tcx> for MultipleReturnTerminators {
    fn is_enabled(&self, sess: &rustc_session::Session) -> bool {
        sess.mir_opt_level() >= 2
    }

    fn run_pass(&self, _: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        let mut to_handle = <Vec<(BasicBlock, SmallVec<_>)>>::new();
        for (bb, bbdata) in body.basic_blocks.iter_enumerated() {
            // Look for returns where, if we lift them into the parents, we can save a block.
            if let TerminatorKind::Return = bbdata.terminator().kind
                && bbdata
                    .statements
                    .iter()
                    .all(|stmt| matches!(stmt.kind, StatementKind::StorageDead(_)))
                && let predecessors = &body.basic_blocks.predecessors()[bb]
                && predecessors.len() >= 2
                && predecessors.iter().all(|pred| {
                    matches!(
                        body.basic_blocks[*pred].terminator().kind,
                        TerminatorKind::Goto { .. },
                    )
                })
            {
                to_handle.push((bb, predecessors.clone()));
            }
        }

        if to_handle.is_empty() {
            return;
        }

        let bbs = body.basic_blocks_mut();
        for (succ, predecessors) in to_handle {
            for pred in predecessors {
                let (pred_block, succ_block) = bbs.pick2_mut(pred, succ);
                pred_block.statements.extend(succ_block.statements.iter().cloned());
                *pred_block.terminator_mut() = succ_block.terminator().clone();
            }
        }
    }

    fn is_required(&self) -> bool {
        false
    }
}
