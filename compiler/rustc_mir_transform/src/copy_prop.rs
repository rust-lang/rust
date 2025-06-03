use rustc_index::IndexSlice;
use rustc_index::bit_set::DenseBitSet;
use rustc_middle::mir::visit::*;
use rustc_middle::mir::*;
use rustc_middle::ty::TyCtxt;
use rustc_mir_dataflow::{Analysis, ResultsCursor};
use tracing::{debug, instrument};

use crate::ssa::{MaybeUninitializedLocals, SsaLocals};

/// Unify locals that copy each other.
///
/// We consider patterns of the form
///   _a = rvalue
///   _b = move? _a
///   _c = move? _a
///   _d = move? _c
/// where each of the locals is only assigned once.
///
/// We want to replace all those locals by `_a` (the "head"), either copied or moved.
pub(super) struct CopyProp;

impl<'tcx> crate::MirPass<'tcx> for CopyProp {
    fn is_enabled(&self, sess: &rustc_session::Session) -> bool {
        sess.mir_opt_level() >= 1
    }

    #[instrument(level = "trace", skip(self, tcx, body))]
    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        debug!(def_id = ?body.source.def_id());

        let typing_env = body.typing_env(tcx);
        let ssa = SsaLocals::new(tcx, body, typing_env);

        debug!(borrowed_locals = ?ssa.borrowed_locals());
        debug!(copy_classes = ?ssa.copy_classes());

        let mut any_replacement = false;
        // Locals that participate in copy propagation either as a source or a destination.
        let mut unified = DenseBitSet::new_empty(body.local_decls.len());
        let mut storage_to_remove = DenseBitSet::new_empty(body.local_decls.len());

        for (local, &head) in ssa.copy_classes().iter_enumerated() {
            if local != head {
                any_replacement = true;
                storage_to_remove.insert(head);
                unified.insert(head);
                unified.insert(local);
            }
        }

        if !any_replacement {
            return;
        }

        // When emitting storage statements, we want to retain the head locals' storage statements,
        // as this enables better optimizations. For each local use location, we mark the head for storage removal
        // only if the head might be uninitialized at that point, or if the local is borrowed
        // (since we cannot easily determine when it's used).
        let storage_to_remove = if tcx.sess.emit_lifetime_markers() {
            storage_to_remove.clear();

            // If the local is borrowed, we cannot easily determine if it is used, so we have to remove the storage statements.
            let borrowed_locals = ssa.borrowed_locals();

            for (local, &head) in ssa.copy_classes().iter_enumerated() {
                if local != head && borrowed_locals.contains(local) {
                    storage_to_remove.insert(head);
                }
            }

            let maybe_uninit = MaybeUninitializedLocals
                .iterate_to_fixpoint(tcx, body, Some("mir_opt::copy_prop"))
                .into_results_cursor(body);

            let mut storage_checker = StorageChecker {
                maybe_uninit,
                copy_classes: ssa.copy_classes(),
                storage_to_remove,
            };

            for (bb, data) in traversal::reachable(body) {
                storage_checker.visit_basic_block_data(bb, data);
            }

            storage_checker.storage_to_remove
        } else {
            // Remove the storage statements of all the head locals.
            storage_to_remove
        };

        debug!(?storage_to_remove);

        Replacer { tcx, copy_classes: ssa.copy_classes(), unified, storage_to_remove }
            .visit_body_preserves_cfg(body);

        crate::simplify::remove_unused_definitions(body);
    }

    fn is_required(&self) -> bool {
        false
    }
}

/// Utility to help performing substitution of `*pattern` by `target`.
struct Replacer<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    unified: DenseBitSet<Local>,
    storage_to_remove: DenseBitSet<Local>,
    copy_classes: &'a IndexSlice<Local, Local>,
}

impl<'tcx> MutVisitor<'tcx> for Replacer<'_, 'tcx> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    #[tracing::instrument(level = "trace", skip(self))]
    fn visit_local(&mut self, local: &mut Local, ctxt: PlaceContext, _: Location) {
        let new_local = self.copy_classes[*local];
        match ctxt {
            // Do not modify the local in storage statements.
            PlaceContext::NonUse(NonUseContext::StorageLive | NonUseContext::StorageDead) => {}
            // We access the value.
            _ => *local = new_local,
        }
    }

    #[tracing::instrument(level = "trace", skip(self))]
    fn visit_operand(&mut self, operand: &mut Operand<'tcx>, loc: Location) {
        if let Operand::Move(place) = *operand
            // A move out of a projection of a copy is equivalent to a copy of the original
            // projection.
            && !place.is_indirect_first_projection()
            && self.unified.contains(place.local)
        {
            *operand = Operand::Copy(place);
        }
        self.super_operand(operand, loc);
    }

    #[tracing::instrument(level = "trace", skip(self))]
    fn visit_statement(&mut self, stmt: &mut Statement<'tcx>, loc: Location) {
        // When removing storage statements, we need to remove both (#107511).
        if let StatementKind::StorageLive(l) | StatementKind::StorageDead(l) = stmt.kind
            && self.storage_to_remove.contains(l)
        {
            stmt.make_nop(true);
        }

        self.super_statement(stmt, loc);

        // Do not leave tautological assignments around.
        if let StatementKind::Assign(box (lhs, ref rhs)) = stmt.kind
            && let Rvalue::Use(Operand::Copy(rhs) | Operand::Move(rhs)) = *rhs
            && lhs == rhs
        {
            stmt.make_nop(true);
        }
    }
}

// Marks heads of copy classes that are maybe uninitialized at the location of a local
// as needing storage statement removal.
struct StorageChecker<'a, 'tcx> {
    maybe_uninit: ResultsCursor<'a, 'tcx, MaybeUninitializedLocals>,
    copy_classes: &'a IndexSlice<Local, Local>,
    storage_to_remove: DenseBitSet<Local>,
}

impl<'a, 'tcx> Visitor<'tcx> for StorageChecker<'a, 'tcx> {
    fn visit_local(&mut self, local: Local, context: PlaceContext, loc: Location) {
        if !context.is_use() {
            return;
        }

        let head = self.copy_classes[local];

        // If the local is the head, or if we already marked it for deletion, we do not need to check it.
        if head == local || self.storage_to_remove.contains(head) {
            return;
        }

        self.maybe_uninit.seek_before_primary_effect(loc);

        if self.maybe_uninit.get().contains(head) {
            debug!(
                ?loc,
                ?context,
                ?local,
                ?head,
                "local's head is maybe uninit at this location, marking head for storage statement removal"
            );
            self.storage_to_remove.insert(head);
        }
    }
}
