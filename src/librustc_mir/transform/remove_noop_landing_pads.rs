// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc::ty::TyCtxt;
use rustc::mir::*;
use rustc_data_structures::bit_set::BitSet;
use transform::{MirPass, MirSource};
use util::patch::MirPatch;

/// A pass that removes no-op landing pads and replaces jumps to them with
/// `None`. This is important because otherwise LLVM generates terrible
/// code for these.
pub struct RemoveNoopLandingPads;

pub fn remove_noop_landing_pads<'a, 'tcx>(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    mir: &mut Mir<'tcx>)
{
    if tcx.sess.no_landing_pads() {
        return
    }
    debug!("remove_noop_landing_pads({:?})", mir);

    RemoveNoopLandingPads.remove_nop_landing_pads(mir)
}

impl MirPass for RemoveNoopLandingPads {
    fn run_pass<'a, 'tcx>(&self,
                          tcx: TyCtxt<'a, 'tcx, 'tcx>,
                          _src: MirSource,
                          mir: &mut Mir<'tcx>) {
        remove_noop_landing_pads(tcx, mir);
    }
}

impl RemoveNoopLandingPads {
    fn is_nop_landing_pad(
        &self,
        bb: BasicBlock,
        mir: &Mir,
        nop_landing_pads: &BitSet<BasicBlock>,
    ) -> bool {
        for stmt in &mir[bb].statements {
            match stmt.kind {
                StatementKind::FakeRead(..) |
                StatementKind::StorageLive(_) |
                StatementKind::StorageDead(_) |
                StatementKind::AscribeUserType(..) |
                StatementKind::Nop => {
                    // These are all nops in a landing pad
                }

                StatementKind::Assign(Place::Local(_), box Rvalue::Use(_)) => {
                    // Writing to a local (e.g. a drop flag) does not
                    // turn a landing pad to a non-nop
                }

                StatementKind::Assign { .. } |
                StatementKind::SetDiscriminant { .. } |
                StatementKind::InlineAsm { .. } |
                StatementKind::Retag { .. } |
                StatementKind::EscapeToRaw { .. } => {
                    return false;
                }
            }
        }

        let terminator = mir[bb].terminator();
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

    fn remove_nop_landing_pads(&self, mir: &mut Mir) {
        // make sure there's a single resume block
        let resume_block = {
            let patch = MirPatch::new(mir);
            let resume_block = patch.resume_block();
            patch.apply(mir);
            resume_block
        };
        debug!("remove_noop_landing_pads: resume block is {:?}", resume_block);

        let mut jumps_folded = 0;
        let mut landing_pads_removed = 0;
        let mut nop_landing_pads = BitSet::new_empty(mir.basic_blocks().len());

        // This is a post-order traversal, so that if A post-dominates B
        // then A will be visited before B.
        let postorder: Vec<_> = traversal::postorder(mir).map(|(bb, _)| bb).collect();
        for bb in postorder {
            debug!("  processing {:?}", bb);
            for target in mir[bb].terminator_mut().successors_mut() {
                if *target != resume_block && nop_landing_pads.contains(*target) {
                    debug!("    folding noop jump to {:?} to resume block", target);
                    *target = resume_block;
                    jumps_folded += 1;
                }
            }

            match mir[bb].terminator_mut().unwind_mut() {
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

            let is_nop_landing_pad = self.is_nop_landing_pad(bb, mir, &nop_landing_pads);
            if is_nop_landing_pad {
                nop_landing_pads.insert(bb);
            }
            debug!("    is_nop_landing_pad({:?}) = {}", bb, is_nop_landing_pad);
        }

        debug!("removed {:?} jumps and {:?} landing pads", jumps_folded, landing_pads_removed);
    }
}
