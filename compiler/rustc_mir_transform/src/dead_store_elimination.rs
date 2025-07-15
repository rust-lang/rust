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

use crate::simplify::UsedInStmtLocals;
use crate::util::is_within_packed;

/// Performs the optimization on the body
///
/// The `borrowed` set must be a `DenseBitSet` of all the locals that are ever borrowed in this
/// body. It can be generated via the [`borrowed_locals`] function.
/// Returns true if any instruction is eliminated.
fn eliminate<'tcx>(tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) -> bool {
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
            if let Some(destination) = MaybeTransitiveLiveLocals::can_be_removed_if_dead(
                &statement.kind,
                &borrowed_locals,
                &debuginfo_locals,
            ) {
                let loc = Location { block: bb, statement_index };
                live.seek_before_primary_effect(loc);
                if !live.get().contains(destination.local) {
                    let drop_debuginfo = !debuginfo_locals.contains(destination.local);
                    // When eliminating a dead statement, we need to address
                    // the debug information for that statement.
                    assert!(
                        drop_debuginfo || statement.kind.as_debuginfo().is_some(),
                        "don't know how to retain the debug information for {:?}",
                        statement.kind
                    );
                    patch.push((loc, drop_debuginfo));
                }
            }
        }
    }

    if patch.is_empty() && call_operands_to_move.is_empty() {
        return false;
    }
    let eliminated = !patch.is_empty();

    let bbs = body.basic_blocks.as_mut_preserves_cfg();
    for (Location { block, statement_index }, drop_debuginfo) in patch {
        bbs[block].statements[statement_index].make_nop(drop_debuginfo);
    }
    for (block, argument_index) in call_operands_to_move {
        let TerminatorKind::Call { ref mut args, .. } = bbs[block].terminator_mut().kind else {
            bug!()
        };
        let arg = &mut args[argument_index].node;
        let Operand::Copy(place) = *arg else { bug!() };
        *arg = Operand::Move(place);
    }

    eliminated
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
        if eliminate(tcx, body) {
            UsedInStmtLocals::new(body).remove_unused_storage_annotations(body);
            for data in body.basic_blocks.as_mut_preserves_cfg() {
                data.strip_nops();
            }
        }
    }

    fn is_required(&self) -> bool {
        false
    }
}
