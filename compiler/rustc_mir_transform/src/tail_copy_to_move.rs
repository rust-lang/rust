//! Rewrite final-use copies before return into moves.
//!
//! # The problem
//!
//! MIR building represents reads of values whose type is `Copy` using
//! `Operand::Copy`, including when such a local is returned. If that local's
//! address has ever been observed, then the local's allocation is semantically
//! valid until its `StorageDead` or function exit. This keeps the local live
//! across `_0 = copy local`, so its live range overlaps with the return place
//! and `MoveElimination` cannot unify the source local with `_0`.
//!
//! # The solution
//!
//! At function return, all local allocations are about to become invalid
//! anyway. After borrowck, this pass can therefore turn a final-use `Copy` into
//! a `Move`, as long as shortening the source local's live range has no
//! observable effect before the return happens. Concretely, between the
//! transformed copy (now a move) and the return, there may only be writes to
//! unborrowed locals, storage markers, nops, and gotos.
//!
//! # The algorithm
//!
//! Start from every `Return` terminator, with `_0` treated as used by the
//! return. Then scan predecessor blocks backward through `Goto` edges, forming
//! a return-tail tree. The scan maintains `used_after`, the set of locals
//! accessed later on that path.
//!
//! A `Copy` operand is rewritten to a `Move` when its base local is not in
//! `used_after`. Then any locals touched by that operand, including
//! index-projection locals, are added to `used_after` before the backward scan
//! continues.
//!
//! The scan stops when accessing an indirect place because that may access any
//! borrowed local, which would make the pass unable to prove any useful final
//! uses. It also stops at writes to borrowed locals, because those can create a
//! new address-observed allocation range whose overlap with an earlier borrowed
//! local must be preserved.

use std::ops::ControlFlow;

use rustc_index::bit_set::DenseBitSet;
use rustc_middle::mir::*;
use rustc_middle::ty::TyCtxt;
use rustc_mir_dataflow::impls::borrowed_locals;

pub(super) struct TailCopyToMove;

impl<'tcx> crate::MirPass<'tcx> for TailCopyToMove {
    fn is_enabled(&self, sess: &rustc_session::Session) -> bool {
        sess.mir_opt_level() >= 2 && sess.opts.unstable_opts.mir_move_elimination
    }

    #[tracing::instrument(level = "trace", skip(self, _tcx, body))]
    fn run_pass(&self, _tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        let borrowed = borrowed_locals(body);
        let predecessors = body.basic_blocks.predecessors().clone();
        let mut stack = Vec::new();

        // A return terminator implicitly uses the return place. Walking
        // backward through assignments records the locals accessed later on
        // this path.
        for (bb, data) in body.basic_blocks.iter_enumerated() {
            if matches!(data.terminator().kind, TerminatorKind::Return) {
                let mut used_after = DenseBitSet::new_empty(body.local_decls.len());
                used_after.insert(RETURN_PLACE);
                stack.push(TailState { block: bb, used_after });
            }
        }

        while let Some(mut state) = stack.pop() {
            // `scan_block` rewrites final-use copies in this block and updates
            // `used_after` to the locals whose allocation is accessed after the
            // block starts. If the block is not pure tail code, this path is
            // done.
            if scan_block(body, state.block, &mut state.used_after, &borrowed).is_break() {
                continue;
            }

            // Continue through predecessor blocks only when the predecessor's
            // terminator is a plain `Goto` to this block. Other terminators are
            // control-flow or effect boundaries.
            let mut first = None;
            for pred in predecessors[state.block].iter().copied() {
                let terminator = body.basic_blocks[pred].terminator();
                if let TerminatorKind::Goto { target } = terminator.kind {
                    debug_assert_eq!(target, state.block);
                    if first.is_none() {
                        first = Some(pred);
                    } else {
                        stack.push(TailState { block: pred, used_after: state.used_after.clone() });
                    }
                }
            }

            // Avoid cloning the bitset for the first predecessor.
            if let Some(pred) = first {
                stack.push(TailState { block: pred, used_after: state.used_after });
            }
        }
    }

    fn is_required(&self) -> bool {
        false
    }
}

struct TailState {
    block: BasicBlock,
    used_after: DenseBitSet<Local>,
}

/// Scan a block backward while the return-tail invariant still holds.
///
/// The invariant is that a whole-local `Copy` can be changed to a `Move` only
/// if this path has no later access to that local's allocation before
/// returning, and no later operation whose observable behavior could depend on
/// ending an address-observed local's allocation early. `used_after` tracks
/// those later local-allocation accesses.
fn scan_block<'tcx>(
    body: &mut Body<'tcx>,
    block: BasicBlock,
    used_after: &mut DenseBitSet<Local>,
    borrowed: &DenseBitSet<Local>,
) -> ControlFlow<()> {
    for statement in body.basic_blocks.as_mut_preserves_cfg()[block].statements.iter_mut().rev() {
        match &mut statement.kind {
            // Under the local lifetime semantics from RFC 3943, `StorageLive`
            // does not allocate, and `StorageDead` has no effect if the local
            // was already freed by a move. These markers therefore do not
            // affect whether a copy can be treated as a final use.
            StatementKind::StorageLive(_) | StatementKind::StorageDead(_) | StatementKind::Nop => {}
            StatementKind::Assign((place, rhs)) => {
                // Accessing an indirect place may touch any borrowed local, so
                // continuing would require treating all borrowed locals as used
                // after this point.
                if place.is_indirect_first_projection() {
                    return ControlFlow::Break(());
                }

                // Writing to a borrowed local can start a new allocation range.
                // Shortening an earlier borrowed local could remove an overlap
                // with that new range.
                if borrowed.contains(place.local) {
                    return ControlFlow::Break(());
                }

                // A destination write accesses the base local, and evaluating
                // the destination may also access projection locals, such as an
                // index.
                record_place_locals(*place, used_after);

                // This pass only models `Use` and `Aggregate` rvalues whose
                // operands are direct. Other rvalues are outside the
                // conservative return-tail shape handled here.
                process_rvalue(rhs, used_after)?;
            }
            StatementKind::SetDiscriminant { place, .. } => {
                // Accessing an indirect place may touch any borrowed local, so
                // continuing would require treating all borrowed locals as used
                // after this point.
                if place.is_indirect_first_projection() {
                    return ControlFlow::Break(());
                }

                // Writing to a borrowed local can start a new allocation range.
                // Shortening an earlier borrowed local could remove an overlap
                // with that new range.
                if borrowed.contains(place.local) {
                    return ControlFlow::Break(());
                }

                // `SetDiscriminant` has a validity invariant on the rest of the
                // place, so treat the base local as accessed along with any
                // projection locals.
                record_place_locals(**place, used_after);
            }
            _ => {
                // Anything else may perform effects or evaluate places in ways
                // this pass does not model, so it is not part of the pure
                // return tail.
                return ControlFlow::Break(());
            }
        }
    }

    ControlFlow::Continue(())
}

/// Records all locals used in a place, including `Index` projections in
/// `used_after`.
fn record_place_locals<'tcx>(place: Place<'tcx>, used_after: &mut DenseBitSet<Local>) {
    for local in place.as_ref().accessed_locals() {
        used_after.insert(local);
    }
}

/// Process the RHS of an assignment in a pure return tail.
fn process_rvalue<'tcx>(
    rvalue: &mut Rvalue<'tcx>,
    used_after: &mut DenseBitSet<Local>,
) -> ControlFlow<()> {
    match rvalue {
        Rvalue::Use(operand, _) => process_operand(operand, used_after),
        Rvalue::Aggregate(_, operands) => {
            // Operands are evaluated left-to-right. We scan them right-to-left
            // so `used_after` includes uses later in the same statement. If an
            // operand accesses an indirect place, only earlier operands and
            // earlier statements are outside the pure tail.
            for operand in operands.iter_mut().rev() {
                process_operand(operand, used_after)?;
            }

            ControlFlow::Continue(())
        }
        _ => {
            // This pass doesn't model other rvalues, so they are not part of
            // the pure return tail.
            ControlFlow::Break(())
        }
    }
}

/// Process one operand in an rvalue.
fn process_operand<'tcx>(
    operand: &mut Operand<'tcx>,
    used_after: &mut DenseBitSet<Local>,
) -> ControlFlow<()> {
    let place = match operand {
        Operand::Copy(place) | Operand::Move(place) if place.is_indirect_first_projection() => {
            // Accessing an indirect place may touch any borrowed local.
            // Continuing would require treating all borrowed locals as used
            // after this point, which would prevent the useful copy-to-move
            // rewrites this pass is looking for.
            return ControlFlow::Break(());
        }
        Operand::Copy(place) => {
            let place = *place;
            // No later operation in the scanned tail accesses this local's
            // allocation, so this copy is a final use on the current return
            // path and can be represented as a move.
            if !used_after.contains(place.local) {
                *operand = Operand::Move(place);
            }
            Some(place)
        }
        Operand::Move(place) => Some(*place),
        Operand::Constant(_) | Operand::RuntimeChecks(_) => None,
    };

    if let Some(place) = place {
        // Reading an operand place accesses its base local, and evaluating its
        // projections may access additional locals, such as the index local in
        // `place[index]`.
        record_place_locals(place, used_after);
    }

    ControlFlow::Continue(())
}
