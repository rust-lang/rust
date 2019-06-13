use rustc::ty::TyCtxt;
use rustc::mir::*;
use rustc_data_structures::bit_set::BitSet;
use crate::transform::{MirPass, MirSource};
use crate::util::patch::MirPatch;

/// A pass that removes noop landing pads and replaces jumps to them with
/// `None`. This is important because otherwise LLVM generates terrible
/// code for these.
pub struct RemoveNoopLandingPads;

pub fn remove_noop_landing_pads<'tcx>(tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
    if tcx.sess.no_landing_pads() {
        return
    }
    debug!("remove_noop_landing_pads({:?})", body);

    RemoveNoopLandingPads.remove_nop_landing_pads(body)
}

impl MirPass for RemoveNoopLandingPads {
    fn run_pass<'tcx>(&self, tcx: TyCtxt<'tcx>, _src: MirSource<'tcx>, body: &mut Body<'tcx>) {
        remove_noop_landing_pads(tcx, body);
    }
}

impl RemoveNoopLandingPads {
    fn is_nop_landing_pad(
        &self,
        bb: BasicBlock,
        body: &Body<'_>,
        nop_landing_pads: &BitSet<BasicBlock>,
    ) -> bool {
        for stmt in &body[bb].statements {
            match stmt.kind {
                StatementKind::FakeRead(..) |
                StatementKind::StorageLive(_) |
                StatementKind::StorageDead(_) |
                StatementKind::AscribeUserType(..) |
                StatementKind::Nop => {
                    // These are all nops in a landing pad
                }

                StatementKind::Assign(Place::Base(PlaceBase::Local(_)), box Rvalue::Use(_)) => {
                    // Writing to a local (e.g., a drop flag) does not
                    // turn a landing pad to a non-nop
                }

                StatementKind::Assign { .. } |
                StatementKind::SetDiscriminant { .. } |
                StatementKind::InlineAsm { .. } |
                StatementKind::Retag { .. } => {
                    return false;
                }
            }
        }

        let terminator = body[bb].terminator();
        match terminator.kind {
            TerminatorKind::Goto { .. } |
            TerminatorKind::Resume |
            TerminatorKind::SwitchInt { .. } |
            TerminatorKind::FalseEdges { .. } |
            TerminatorKind::FalseUnwind { .. } => {
                terminator.successors().all(|&succ| {
                    nop_landing_pads.contains(succ)
                })
            },
            TerminatorKind::GeneratorDrop |
            TerminatorKind::Yield { .. } |
            TerminatorKind::Return |
            TerminatorKind::Abort |
            TerminatorKind::Unreachable |
            TerminatorKind::Call { .. } |
            TerminatorKind::Assert { .. } |
            TerminatorKind::DropAndReplace { .. } |
            TerminatorKind::Drop { .. } => {
                false
            }
        }
    }

    fn remove_nop_landing_pads(&self, body: &mut Body<'_>) {
        // make sure there's a single resume block
        let resume_block = {
            let patch = MirPatch::new(body);
            let resume_block = patch.resume_block();
            patch.apply(body);
            resume_block
        };
        debug!("remove_noop_landing_pads: resume block is {:?}", resume_block);

        let mut jumps_folded = 0;
        let mut landing_pads_removed = 0;
        let mut nop_landing_pads = BitSet::new_empty(body.basic_blocks().len());

        // This is a post-order traversal, so that if A post-dominates B
        // then A will be visited before B.
        let postorder: Vec<_> = traversal::postorder(body).map(|(bb, _)| bb).collect();
        for bb in postorder {
            debug!("  processing {:?}", bb);
            for target in body[bb].terminator_mut().successors_mut() {
                if *target != resume_block && nop_landing_pads.contains(*target) {
                    debug!("    folding noop jump to {:?} to resume block", target);
                    *target = resume_block;
                    jumps_folded += 1;
                }
            }

            match body[bb].terminator_mut().unwind_mut() {
                Some(unwind) => {
                    if *unwind == Some(resume_block) {
                        debug!("    removing noop landing pad");
                        jumps_folded -= 1;
                        landing_pads_removed += 1;
                        *unwind = None;
                    }
                }
                _ => {}
            }

            let is_nop_landing_pad = self.is_nop_landing_pad(bb, body, &nop_landing_pads);
            if is_nop_landing_pad {
                nop_landing_pads.insert(bb);
            }
            debug!("    is_nop_landing_pad({:?}) = {}", bb, is_nop_landing_pad);
        }

        debug!("removed {:?} jumps and {:?} landing pads", jumps_folded, landing_pads_removed);
    }
}
