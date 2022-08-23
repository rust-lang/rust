//! A pass that propagates the unreachable terminator of a block to its predecessors
//! when all of their successors are unreachable. This is achieved through a
//! post-order traversal of the blocks.

use crate::simplify;
use crate::MirPass;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_middle::mir::*;
use rustc_middle::ty::TyCtxt;

pub struct UnreachablePropagation;

impl MirPass<'_> for UnreachablePropagation {
    fn is_enabled(&self, sess: &rustc_session::Session) -> bool {
        // Enable only under -Zmir-opt-level=2 as this can make programs less debuggable.
        sess.mir_opt_level() >= 2
    }

    fn run_pass<'tcx>(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        let mut unreachable_blocks = FxHashSet::default();
        let mut replacements = FxHashMap::default();

        for (bb, bb_data) in traversal::postorder(body) {
            let terminator = bb_data.terminator();
            if terminator.kind == TerminatorKind::Unreachable {
                unreachable_blocks.insert(bb);
            } else {
                let is_unreachable = |succ: BasicBlock| unreachable_blocks.contains(&succ);
                let terminator_kind_opt = remove_successors(&terminator.kind, is_unreachable);

                if let Some(terminator_kind) = terminator_kind_opt {
                    if terminator_kind == TerminatorKind::Unreachable {
                        unreachable_blocks.insert(bb);
                    }
                    replacements.insert(bb, terminator_kind);
                }
            }
        }

        // We do want do keep some unreachable blocks, but make them empty.
        for bb in unreachable_blocks {
            if !tcx.consider_optimizing(|| {
                format!("UnreachablePropagation {:?} ", body.source.def_id())
            }) {
                break;
            }

            body.basic_blocks_mut()[bb].statements.clear();
        }

        let replaced = !replacements.is_empty();

        for (bb, terminator_kind) in replacements {
            if !tcx.consider_optimizing(|| {
                format!("UnreachablePropagation {:?} ", body.source.def_id())
            }) {
                break;
            }

            body.basic_blocks_mut()[bb].terminator_mut().kind = terminator_kind;
        }

        if replaced {
            simplify::remove_dead_blocks(tcx, body);
        }
    }
}

fn remove_successors<'tcx, F>(
    terminator_kind: &TerminatorKind<'tcx>,
    is_unreachable: F,
) -> Option<TerminatorKind<'tcx>>
where
    F: Fn(BasicBlock) -> bool,
{
    let terminator = match terminator_kind {
        // This will unconditionally run into an unreachable and is therefore unreachable as well.
        TerminatorKind::Goto { target } if is_unreachable(*target) => TerminatorKind::Unreachable,
        TerminatorKind::SwitchInt { targets, discr, switch_ty } => {
            let otherwise = targets.otherwise();

            // If all targets are unreachable, we can be unreachable as well.
            if targets.all_targets().iter().all(|bb| is_unreachable(*bb)) {
                TerminatorKind::Unreachable
            } else if is_unreachable(otherwise) {
                // If there are multiple targets, don't delete unreachable branches (like an unreachable otherwise)
                // unless otherwise is unrachable, in which case deleting a normal branch causes it to be merged with
                // the otherwise, keeping its unreachable.
                // This looses information about reachability causing worse codegen.
                // For example (see src/test/codegen/match-optimizes-away.rs)
                //
                // pub enum Two { A, B }
                // pub fn identity(x: Two) -> Two {
                //     match x {
                //         Two::A => Two::A,
                //         Two::B => Two::B,
                //     }
                // }
                //
                // This generates a `switchInt() -> [0: 0, 1: 1, otherwise: unreachable]`, which allows us or LLVM to
                // turn it into just `x` later. Without the unreachable, such a transformation would be illegal.
                // If the otherwise branch is unreachable, we can delete all other unreacahble targets, as they will
                // still point to the unreachable and therefore not lose reachability information.
                let reachable_iter = targets.iter().filter(|(_, bb)| !is_unreachable(*bb));

                let new_targets = SwitchTargets::new(reachable_iter, otherwise);

                // No unreachable branches were removed.
                if new_targets.all_targets().len() == targets.all_targets().len() {
                    return None;
                }

                TerminatorKind::SwitchInt {
                    discr: discr.clone(),
                    switch_ty: *switch_ty,
                    targets: new_targets,
                }
            } else {
                // If the otherwise branch is reachable, we don't want to delete any unreachable branches.
                return None;
            }
        }
        _ => return None,
    };
    Some(terminator)
}
