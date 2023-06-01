//! A pass that duplicates switch-terminated blocks
//! into a new copy for each predecessor, provided
//! the predecessor sets the value being switched
//! over to a constant.
//!
//! The purpose of this pass is to help constant
//! propagation passes to simplify the switch terminator
//! of the copied blocks into gotos when some predecessors
//! statically determine the output of switches.
//!
//! ```text
//!     x = 12 ---              ---> something
//!               \            / 12
//!                --> switch x
//!               /            \ otherwise
//!     x = y  ---              ---> something else
//! ```
//! becomes
//! ```text
//!     x = 12 ---> switch x ------> something
//!                          \ / 12
//!                           X
//!                          / \ otherwise
//!     x = y  ---> switch x ------> something else
//! ```
//! so it can hopefully later be turned by another pass into
//! ```text
//!     x = 12 --------------------> something
//!                            / 12
//!                           /
//!                          /   otherwise
//!     x = y  ---- switch x ------> something else
//! ```
//!
//! This optimization is meant to cover simple cases
//! like `?` desugaring. For now, it thus focuses on
//! simplicity rather than completeness (it notably
//! sometimes duplicates abusively).

use crate::MirPass;
use rustc_middle::mir::*;
use rustc_middle::ty::TyCtxt;
use smallvec::SmallVec;

pub struct SeparateConstSwitch;

impl<'tcx> MirPass<'tcx> for SeparateConstSwitch {
    fn is_enabled(&self, sess: &rustc_session::Session) -> bool {
        sess.mir_opt_level() >= 2
    }

    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        // If execution did something, applying a simplification layer
        // helps later passes optimize the copy away.
        if separate_const_switch(body) > 0 {
            super::simplify::simplify_cfg(tcx, body);
        }
    }
}

/// Returns the amount of blocks that were duplicated
pub fn separate_const_switch(body: &mut Body<'_>) -> usize {
    let mut new_blocks: SmallVec<[(BasicBlock, BasicBlock); 6]> = SmallVec::new();
    let predecessors = body.basic_blocks.predecessors();
    'block_iter: for (block_id, block) in body.basic_blocks.iter_enumerated() {
        if let TerminatorKind::SwitchInt {
            discr: Operand::Copy(switch_place) | Operand::Move(switch_place),
            ..
        } = block.terminator().kind
        {
            // If the block is on an unwind path, do not
            // apply the optimization as unwind paths
            // rely on a unique parent invariant
            if block.is_cleanup {
                continue 'block_iter;
            }

            // If the block has fewer than 2 predecessors, ignore it
            // we could maybe chain blocks that have exactly one
            // predecessor, but for now we ignore that
            if predecessors[block_id].len() < 2 {
                continue 'block_iter;
            }

            // First, let's find a non-const place
            // that determines the result of the switch
            if let Some(switch_place) = find_determining_place(switch_place, block) {
                // We now have an input place for which it would
                // be interesting if predecessors assigned it from a const

                let mut predecessors_left = predecessors[block_id].len();
                'predec_iter: for predecessor_id in predecessors[block_id].iter().copied() {
                    let predecessor = &body.basic_blocks[predecessor_id];

                    // First we make sure the predecessor jumps
                    // in a reasonable way
                    match &predecessor.terminator().kind {
                        // The following terminators are
                        // unconditionally valid
                        TerminatorKind::Goto { .. } | TerminatorKind::SwitchInt { .. } => {}

                        TerminatorKind::FalseEdge { real_target, .. } => {
                            if *real_target != block_id {
                                continue 'predec_iter;
                            }
                        }

                        // The following terminators are not allowed
                        TerminatorKind::Resume
                        | TerminatorKind::Drop { .. }
                        | TerminatorKind::Call { .. }
                        | TerminatorKind::Assert { .. }
                        | TerminatorKind::FalseUnwind { .. }
                        | TerminatorKind::Yield { .. }
                        | TerminatorKind::Terminate
                        | TerminatorKind::Return
                        | TerminatorKind::Unreachable
                        | TerminatorKind::InlineAsm { .. }
                        | TerminatorKind::GeneratorDrop => {
                            continue 'predec_iter;
                        }
                    }

                    if is_likely_const(switch_place, predecessor) {
                        new_blocks.push((predecessor_id, block_id));
                        predecessors_left -= 1;
                        if predecessors_left < 2 {
                            // If the original block only has one predecessor left,
                            // we have nothing left to do
                            break 'predec_iter;
                        }
                    }
                }
            }
        }
    }

    // Once the analysis is done, perform the duplication
    let body_span = body.span;
    let copied_blocks = new_blocks.len();
    let blocks = body.basic_blocks_mut();
    for (pred_id, target_id) in new_blocks {
        let new_block = blocks[target_id].clone();
        let new_block_id = blocks.push(new_block);
        let terminator = blocks[pred_id].terminator_mut();

        match terminator.kind {
            TerminatorKind::Goto { ref mut target } => {
                *target = new_block_id;
            }

            TerminatorKind::FalseEdge { ref mut real_target, .. } => {
                if *real_target == target_id {
                    *real_target = new_block_id;
                }
            }

            TerminatorKind::SwitchInt { ref mut targets, .. } => {
                targets.all_targets_mut().iter_mut().for_each(|x| {
                    if *x == target_id {
                        *x = new_block_id;
                    }
                });
            }

            TerminatorKind::Resume
            | TerminatorKind::Terminate
            | TerminatorKind::Return
            | TerminatorKind::Unreachable
            | TerminatorKind::GeneratorDrop
            | TerminatorKind::Assert { .. }
            | TerminatorKind::FalseUnwind { .. }
            | TerminatorKind::Drop { .. }
            | TerminatorKind::Call { .. }
            | TerminatorKind::InlineAsm { .. }
            | TerminatorKind::Yield { .. } => {
                span_bug!(
                    body_span,
                    "basic block terminator had unexpected kind {:?}",
                    &terminator.kind
                )
            }
        }
    }

    copied_blocks
}

/// This function describes a rough heuristic guessing
/// whether a place is last set with a const within the block.
/// Notably, it will be overly pessimistic in cases that are already
/// not handled by `separate_const_switch`.
fn is_likely_const<'tcx>(mut tracked_place: Place<'tcx>, block: &BasicBlockData<'tcx>) -> bool {
    for statement in block.statements.iter().rev() {
        match &statement.kind {
            StatementKind::Assign(assign) => {
                if assign.0 == tracked_place {
                    match assign.1 {
                        // These rvalues are definitely constant
                        Rvalue::Use(Operand::Constant(_))
                        | Rvalue::Ref(_, _, _)
                        | Rvalue::AddressOf(_, _)
                        | Rvalue::Cast(_, Operand::Constant(_), _)
                        | Rvalue::NullaryOp(_, _)
                        | Rvalue::ShallowInitBox(_, _)
                        | Rvalue::UnaryOp(_, Operand::Constant(_)) => return true,

                        // These rvalues make things ambiguous
                        Rvalue::Repeat(_, _)
                        | Rvalue::ThreadLocalRef(_)
                        | Rvalue::Len(_)
                        | Rvalue::BinaryOp(_, _)
                        | Rvalue::CheckedBinaryOp(_, _)
                        | Rvalue::Aggregate(_, _) => return false,

                        // These rvalues move the place to track
                        Rvalue::Cast(_, Operand::Copy(place) | Operand::Move(place), _)
                        | Rvalue::Use(Operand::Copy(place) | Operand::Move(place))
                        | Rvalue::CopyForDeref(place)
                        | Rvalue::UnaryOp(_, Operand::Copy(place) | Operand::Move(place))
                        | Rvalue::Discriminant(place) => tracked_place = place,
                    }
                }
            }

            // If the discriminant is set, it is always set
            // as a constant, so the job is done.
            // As we are **ignoring projections**, if the place
            // we are tracking sees its discriminant be set,
            // that means we had to be tracking the discriminant
            // specifically (as it is impossible to switch over
            // an enum directly, and if we were switching over
            // its content, we would have had to at least cast it to
            // some variant first)
            StatementKind::SetDiscriminant { place, .. } => {
                if **place == tracked_place {
                    return true;
                }
            }

            // These statements have no influence on the place
            // we are interested in
            StatementKind::FakeRead(_)
            | StatementKind::Deinit(_)
            | StatementKind::StorageLive(_)
            | StatementKind::Retag(_, _)
            | StatementKind::AscribeUserType(_, _)
            | StatementKind::PlaceMention(..)
            | StatementKind::Coverage(_)
            | StatementKind::StorageDead(_)
            | StatementKind::Intrinsic(_)
            | StatementKind::ConstEvalCounter
            | StatementKind::Nop => {}
        }
    }

    // If no good reason for the place to be const is found,
    // give up. We could maybe go up predecessors, but in
    // most cases giving up now should be sufficient.
    false
}

/// Finds a unique place that entirely determines the value
/// of `switch_place`, if it exists. This is only a heuristic.
/// Ideally we would like to track multiple determining places
/// for some edge cases, but one is enough for a lot of situations.
fn find_determining_place<'tcx>(
    mut switch_place: Place<'tcx>,
    block: &BasicBlockData<'tcx>,
) -> Option<Place<'tcx>> {
    for statement in block.statements.iter().rev() {
        match &statement.kind {
            StatementKind::Assign(op) => {
                if op.0 != switch_place {
                    continue;
                }

                match op.1 {
                    // The following rvalues move the place
                    // that may be const in the predecessor
                    Rvalue::Use(Operand::Move(new) | Operand::Copy(new))
                    | Rvalue::UnaryOp(_, Operand::Copy(new) | Operand::Move(new))
                    | Rvalue::CopyForDeref(new)
                    | Rvalue::Cast(_, Operand::Move(new) | Operand::Copy(new), _)
                    | Rvalue::Repeat(Operand::Move(new) | Operand::Copy(new), _)
                    | Rvalue::Discriminant(new)
                    => switch_place = new,

                    // The following rvalues might still make the block
                    // be valid but for now we reject them
                    Rvalue::Len(_)
                    | Rvalue::Ref(_, _, _)
                    | Rvalue::BinaryOp(_, _)
                    | Rvalue::CheckedBinaryOp(_, _)
                    | Rvalue::Aggregate(_, _)

                    // The following rvalues definitely mean we cannot
                    // or should not apply this optimization
                    | Rvalue::Use(Operand::Constant(_))
                    | Rvalue::Repeat(Operand::Constant(_), _)
                    | Rvalue::ThreadLocalRef(_)
                    | Rvalue::AddressOf(_, _)
                    | Rvalue::NullaryOp(_, _)
                    | Rvalue::ShallowInitBox(_, _)
                    | Rvalue::UnaryOp(_, Operand::Constant(_))
                    | Rvalue::Cast(_, Operand::Constant(_), _) => return None,
                }
            }

            // These statements have no influence on the place
            // we are interested in
            StatementKind::FakeRead(_)
            | StatementKind::Deinit(_)
            | StatementKind::StorageLive(_)
            | StatementKind::StorageDead(_)
            | StatementKind::Retag(_, _)
            | StatementKind::AscribeUserType(_, _)
            | StatementKind::PlaceMention(..)
            | StatementKind::Coverage(_)
            | StatementKind::Intrinsic(_)
            | StatementKind::ConstEvalCounter
            | StatementKind::Nop => {}

            // If the discriminant is set, it is always set
            // as a constant, so the job is already done.
            // As we are **ignoring projections**, if the place
            // we are tracking sees its discriminant be set,
            // that means we had to be tracking the discriminant
            // specifically (as it is impossible to switch over
            // an enum directly, and if we were switching over
            // its content, we would have had to at least cast it to
            // some variant first)
            StatementKind::SetDiscriminant { place, .. } => {
                if **place == switch_place {
                    return None;
                }
            }
        }
    }

    Some(switch_place)
}
