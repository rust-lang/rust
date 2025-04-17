//! This module implements a dead store elimination (DSE) routine.
//!
//! This transformation was written specifically for the needs of dest prop. Although it is
//! perfectly sound to use it in any context that might need it, its behavior should not be changed
//! without analyzing the interaction this will have with dest prop. Specifically, in addition to
//! the soundness of this pass in general, dest prop needs it to satisfy two additional conditions:
//!
//!  1. It's idempotent, meaning that running this pass a second time immediately after running it a
//!     first time will not cause any further changes.
//!  2. This idempotence persists across dest prop's main transform, in other words inserting any
//!     number of iterations of dest prop between the first and second application of this transform
//!     will still not cause any further changes.
//!

use rustc_middle::bug;
use rustc_middle::mir::visit::Visitor;
use rustc_middle::mir::*;
use rustc_middle::ty::TyCtxt;
use rustc_mir_dataflow::Analysis;
use rustc_mir_dataflow::debuginfo::debuginfo_locals;
use rustc_mir_dataflow::impls::{
    LivenessTransferFunction, MaybeTransitiveLiveLocals, borrowed_locals,
};

use crate::util::is_within_packed;

/// Performs the optimization on the body
///
/// The `borrowed` set must be a `DenseBitSet` of all the locals that are ever borrowed in this
/// body. It can be generated via the [`borrowed_locals`] function.
fn eliminate<'tcx>(tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
    let borrowed_locals = borrowed_locals(body);

    // If the user requests complete debuginfo, mark the locals that appear in it as live, so
    // we don't remove assignments to them.
    let debuginfo_locals = debuginfo_locals(body);

    let mut live = MaybeTransitiveLiveLocals::new(&borrowed_locals, &debuginfo_locals)
        .iterate_to_fixpoint(tcx, body, None)
        .into_results_cursor(body);

    // For blocks with a call terminator, if an argument copy can be turned into a move,
    // record it as (block, argument index).
    let mut call_operands_to_move = Vec::new();
    let mut patch = Vec::new();

    for (bb, bb_data) in traversal::preorder(body) {
        if let TerminatorKind::Call { ref args, .. } = bb_data.terminator().kind {
            let loc = Location { block: bb, statement_index: bb_data.statements.len() };

            // Position ourselves between the evaluation of `args` and the write to `destination`.
            live.seek_to_block_end(bb);
            let mut state = live.get().clone();

            for (index, arg) in args.iter().map(|a| &a.node).enumerate().rev() {
                if let Operand::Copy(place) = *arg
                    && !place.is_indirect()
                    // Do not skip the transformation if the local is in debuginfo, as we do
                    // not really lose any information for this purpose.
                    && !borrowed_locals.contains(place.local)
                    && !state.contains(place.local)
                    // If `place` is a projection of a disaligned field in a packed ADT,
                    // the move may be codegened as a pointer to that field.
                    // Using that disaligned pointer may trigger UB in the callee,
                    // so do nothing.
                    && is_within_packed(tcx, body, place).is_none()
                {
                    call_operands_to_move.push((bb, index));
                }

                // Account that `arg` is read from, so we don't promote another argument to a move.
                LivenessTransferFunction(&mut state).visit_operand(arg, loc);
            }
        }

        for (statement_index, statement) in bb_data.statements.iter().enumerate().rev() {
            let loc = Location { block: bb, statement_index };
            match &statement.kind {
                StatementKind::Assign(box (place, rvalue)) => {
                    if rvalue.is_safe_to_remove()
                        && !place.is_indirect()
                        && !borrowed_locals.contains(place.local)
                        && (!debuginfo_locals.contains(place.local)
                            || (place.as_local().is_some() && matches!(rvalue, Rvalue::Ref(..))))
                    {
                        live.seek_before_primary_effect(loc);
                        if !live.get().contains(place.local) {
                            patch.push(loc);
                        }
                    }
                }
                StatementKind::SetDiscriminant { place: box place, .. }
                | StatementKind::Deinit(box place) => {
                    if !place.is_indirect()
                        && !borrowed_locals.contains(place.local)
                        && !debuginfo_locals.contains(place.local)
                    {
                        live.seek_before_primary_effect(loc);
                        if !live.get().contains(place.local) {
                            patch.push(loc);
                        }
                    }
                }
                StatementKind::Retag(_, _)
                | StatementKind::StorageLive(_)
                | StatementKind::StorageDead(_)
                | StatementKind::Coverage(_)
                | StatementKind::Intrinsic(_)
                | StatementKind::ConstEvalCounter
                | StatementKind::PlaceMention(_)
                | StatementKind::BackwardIncompatibleDropHint { .. }
                | StatementKind::Nop(_) => {}

                StatementKind::FakeRead(_) | StatementKind::AscribeUserType(_, _) => {
                    bug!("{:?} not found in this MIR phase!", statement.kind)
                }
            }
        }
    }

    if patch.is_empty() && call_operands_to_move.is_empty() {
        return;
    }

    let bbs = body.basic_blocks.as_mut_preserves_cfg();
    for Location { block, statement_index } in patch {
        let save_stmt = match bbs[block].statements[statement_index].kind {
            StatementKind::Assign(box (place, _)) => debuginfo_locals.contains(place.local),
            StatementKind::SetDiscriminant { .. }
            | StatementKind::Deinit(_)
            | StatementKind::Retag(_, _)
            | StatementKind::StorageLive(_)
            | StatementKind::StorageDead(_)
            | StatementKind::Coverage(_)
            | StatementKind::Intrinsic(_)
            | StatementKind::ConstEvalCounter
            | StatementKind::PlaceMention(_)
            | StatementKind::BackwardIncompatibleDropHint { .. }
            | StatementKind::FakeRead(_)
            | StatementKind::AscribeUserType(_, _)
            | StatementKind::Nop(_) => false,
        };
        bbs[block].statements[statement_index].make_nop(save_stmt);
    }
    for (block, argument_index) in call_operands_to_move {
        let TerminatorKind::Call { ref mut args, .. } = bbs[block].terminator_mut().kind else {
            bug!()
        };
        let arg = &mut args[argument_index].node;
        let Operand::Copy(place) = *arg else { bug!() };
        *arg = Operand::Move(place);
    }
}

pub(super) enum DeadStoreElimination {
    Initial,
    Final,
}

impl<'tcx> crate::MirPass<'tcx> for DeadStoreElimination {
    fn name(&self) -> &'static str {
        match self {
            DeadStoreElimination::Initial => "DeadStoreElimination-initial",
            DeadStoreElimination::Final => "DeadStoreElimination-final",
        }
    }

    fn is_enabled(&self, sess: &rustc_session::Session) -> bool {
        sess.mir_opt_level() >= 2
    }

    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        eliminate(tcx, body);
    }

    fn is_required(&self) -> bool {
        false
    }
}
