use rustc_index::bit_set::DenseBitSet;
use rustc_middle::mir::*;
use rustc_middle::ty::TyCtxt;
use tracing::{debug, instrument};

use crate::patch::MirPatch;

/// A pass that removes noop landing pads and replaces jumps to them with
/// `UnwindAction::Continue`. This is important because otherwise LLVM generates
/// terrible code for these.
pub(super) struct RemoveNoopLandingPads;

impl<'tcx> crate::MirPass<'tcx> for RemoveNoopLandingPads {
    fn is_enabled(&self, sess: &rustc_session::Session) -> bool {
        sess.panic_strategy().unwinds()
    }

    #[instrument(level = "debug", skip(self, _tcx, body))]
    fn run_pass(&self, _tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        let def_id = body.source.def_id();
        debug!(?def_id);

        // Skip the pass if there are no blocks with a resume terminator.
        let has_resume = body
            .basic_blocks
            .iter_enumerated()
            .any(|(_bb, block)| matches!(block.terminator().kind, TerminatorKind::UnwindResume));
        if !has_resume {
            debug!("no resume block in MIR");
            return;
        }

        let mut nop_landing_pads = DenseBitSet::new_empty(body.basic_blocks.len());

        // This is a post-order traversal, so that if A post-dominates B
        // then A will be visited before B.
        for (bb, bbdata) in traversal::postorder(body) {
            let is_nop_landing_pad = self.is_nop_landing_pad(bbdata, &nop_landing_pads);
            debug!("is_nop_landing_pad({bb:?}) = {is_nop_landing_pad}");
            if is_nop_landing_pad {
                nop_landing_pads.insert(bb);
            }
        }

        if nop_landing_pads.is_empty() {
            debug!("no nop landing pads in MIR");
            return;
        }

        // make sure there's a resume block without any statements
        let resume_block = {
            let mut patch = MirPatch::new(body);
            let resume_block = patch.resume_block();
            patch.apply(body);
            resume_block
        };
        debug!(?resume_block);

        let basic_blocks = body.basic_blocks.as_mut();
        for (bb, bbdata) in basic_blocks.iter_enumerated_mut() {
            debug!("processing {:?}", bb);

            if let Some(unwind) = bbdata.terminator_mut().unwind_mut()
                && let UnwindAction::Cleanup(unwind_bb) = *unwind
                && nop_landing_pads.contains(unwind_bb)
            {
                debug!("    removing noop landing pad");
                *unwind = UnwindAction::Continue;
            }

            bbdata.terminator_mut().successors_mut(|target| {
                if *target != resume_block && nop_landing_pads.contains(*target) {
                    debug!("    folding noop jump to {:?} to resume block", target);
                    *target = resume_block;
                }
            });
        }
    }

    fn is_required(&self) -> bool {
        true
    }
}

impl RemoveNoopLandingPads {
    fn is_nop_landing_pad(
        &self,
        bbdata: &BasicBlockData<'_>,
        nop_landing_pads: &DenseBitSet<BasicBlock>,
    ) -> bool {
        for stmt in &bbdata.statements {
            match &stmt.kind {
                StatementKind::FakeRead(..)
                | StatementKind::StorageLive(_)
                | StatementKind::StorageDead(_)
                | StatementKind::PlaceMention(..)
                | StatementKind::AscribeUserType(..)
                | StatementKind::Coverage(..)
                | StatementKind::ConstEvalCounter
                | StatementKind::BackwardIncompatibleDropHint { .. }
                | StatementKind::Nop => {
                    // These are all noops in a landing pad
                }

                StatementKind::Assign(box (place, Rvalue::Use(_) | Rvalue::Discriminant(_))) => {
                    if place.as_local().is_some() {
                        // Writing to a local (e.g., a drop flag) does not
                        // turn a landing pad to a non-nop
                    } else {
                        return false;
                    }
                }

                StatementKind::Assign { .. }
                | StatementKind::SetDiscriminant { .. }
                | StatementKind::Deinit(..)
                | StatementKind::Intrinsic(..)
                | StatementKind::Retag { .. } => {
                    return false;
                }
            }
        }

        let terminator = bbdata.terminator();
        match terminator.kind {
            TerminatorKind::Goto { .. }
            | TerminatorKind::UnwindResume
            | TerminatorKind::SwitchInt { .. }
            | TerminatorKind::FalseEdge { .. }
            | TerminatorKind::FalseUnwind { .. } => {
                terminator.successors().all(|succ| nop_landing_pads.contains(succ))
            }
            TerminatorKind::CoroutineDrop
            | TerminatorKind::Yield { .. }
            | TerminatorKind::Return
            | TerminatorKind::UnwindTerminate(_)
            | TerminatorKind::Unreachable
            | TerminatorKind::Call { .. }
            | TerminatorKind::TailCall { .. }
            | TerminatorKind::Assert { .. }
            | TerminatorKind::Drop { .. }
            | TerminatorKind::InlineAsm { .. } => false,
        }
    }
}
