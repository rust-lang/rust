use rustc_index::bit_set::DenseBitSet;
use rustc_middle::mir::*;
use rustc_middle::ty;
use rustc_middle::ty::{Instance, TyCtxt};
use rustc_target::spec::PanicStrategy;
use tracing::debug;

use crate::patch::MirPatch;

/// A pass that removes noop landing pads and replaces jumps to them with
/// `UnwindAction::Continue`. This is important because otherwise LLVM generates
/// terrible code for these.
pub(super) struct RemoveNoopLandingPads;

impl<'tcx> crate::MirPass<'tcx> for RemoveNoopLandingPads {
    fn is_enabled(&self, sess: &rustc_session::Session) -> bool {
        sess.panic_strategy() != PanicStrategy::Abort
    }

    fn run_pass(&self, _tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        let def_id = body.source.def_id();
        debug!(?def_id);

        // Skip the pass if there are no blocks with a resume terminator.
        let has_resume = body
            .basic_blocks
            .iter_enumerated()
            .any(|(_bb, block)| matches!(block.terminator().kind, TerminatorKind::UnwindResume));
        if !has_resume {
            debug!("remove_noop_landing_pads: no resume block in MIR");
            return;
        }

        // make sure there's a resume block without any statements
        let resume_block = {
            let mut patch = MirPatch::new(body);
            let resume_block = patch.resume_block();
            patch.apply(body);
            resume_block
        };
        debug!("remove_noop_landing_pads: resume block is {:?}", resume_block);

        let mut jumps_folded = 0;
        let mut landing_pads_removed = 0;
        let nop_landing_pads = find_noop_landing_pads(body, None);

        // This is a post-order traversal, so that if A post-dominates B
        // then A will be visited before B.
        for (bb, block) in body.basic_blocks_mut().iter_enumerated_mut() {
            debug!("  processing {:?}", bb);
            if let Some(unwind) = block.terminator_mut().unwind_mut() {
                if let UnwindAction::Cleanup(unwind_bb) = *unwind {
                    if nop_landing_pads.contains(unwind_bb) {
                        debug!("    removing noop landing pad");
                        landing_pads_removed += 1;
                        *unwind = UnwindAction::Continue;
                    }
                }
            }

            block.terminator_mut().successors_mut(|target| {
                if *target != resume_block && nop_landing_pads.contains(*target) {
                    debug!("    folding noop jump to {:?} to resume block", target);
                    *target = resume_block;
                    jumps_folded += 1;
                }
            });
        }

        debug!("removed {:?} jumps and {:?} landing pads", jumps_folded, landing_pads_removed);
    }

    fn is_required(&self) -> bool {
        true
    }
}

/// This provides extra information that allows further analysis.
///
/// Used by rustc_codegen_ssa.
pub struct ExtraInfo<'tcx> {
    pub tcx: TyCtxt<'tcx>,
    pub instance: Instance<'tcx>,
    pub typing_env: ty::TypingEnv<'tcx>,
}

pub fn find_noop_landing_pads<'tcx>(
    body: &Body<'tcx>,
    extra: Option<ExtraInfo<'tcx>>,
) -> DenseBitSet<BasicBlock> {
    let mut nop_landing_pads = DenseBitSet::new_empty(body.basic_blocks.len());

    // This is a post-order traversal, so that if A post-dominates B
    // then A will be visited before B.
    let postorder: Vec<_> = traversal::postorder(body).map(|(bb, _)| bb).collect();
    for bb in postorder {
        let is_nop_landing_pad = is_nop_landing_pad(bb, body, &nop_landing_pads, extra.as_ref());
        if is_nop_landing_pad {
            nop_landing_pads.insert(bb);
        }
        debug!("    is_nop_landing_pad({:?}) = {}", bb, is_nop_landing_pad);
    }

    nop_landing_pads
}

fn is_nop_landing_pad<'tcx>(
    bb: BasicBlock,
    body: &Body<'tcx>,
    nop_landing_pads: &DenseBitSet<BasicBlock>,
    extra: Option<&ExtraInfo<'tcx>>,
) -> bool {
    for stmt in &body[bb].statements {
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

    let terminator = body[bb].terminator();
    match terminator.kind {
        TerminatorKind::Goto { .. }
        | TerminatorKind::UnwindResume
        | TerminatorKind::SwitchInt { .. }
        | TerminatorKind::FalseEdge { .. }
        | TerminatorKind::FalseUnwind { .. } => {
            terminator.successors().all(|succ| nop_landing_pads.contains(succ))
        }

        TerminatorKind::Drop { place, .. } => {
            if let Some(extra) = extra {
                let ty = place.ty(body, extra.tcx).ty;
                debug!("monomorphize: instance={:?}", extra.instance);
                let ty = extra.instance.instantiate_mir_and_normalize_erasing_regions(
                    extra.tcx,
                    extra.typing_env,
                    ty::EarlyBinder::bind(ty),
                );
                let drop_fn = Instance::resolve_drop_in_place(extra.tcx, ty);
                if let ty::InstanceKind::DropGlue(_, None) = drop_fn.def {
                    // no need to drop anything, if all of our successors are also no-op then we
                    // can be skipped.
                    return terminator.successors().all(|succ| nop_landing_pads.contains(succ));
                }
            }

            false
        }

        TerminatorKind::CoroutineDrop
        | TerminatorKind::Yield { .. }
        | TerminatorKind::Return
        | TerminatorKind::UnwindTerminate(_)
        | TerminatorKind::Call { .. }
        | TerminatorKind::TailCall { .. }
        | TerminatorKind::Unreachable
        | TerminatorKind::Assert { .. }
        | TerminatorKind::InlineAsm { .. } => false,
    }
}
