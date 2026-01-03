use rustc_index::IndexSlice;
use rustc_index::bit_set::DenseBitSet;
use rustc_middle::mir::visit::*;
use rustc_middle::mir::*;
use rustc_middle::ty::TyCtxt;
use tracing::{debug, instrument};

use crate::ssa::SsaLocals;

/// Unify locals that copy each other.
///
/// We consider patterns of the form
///   _a = rvalue
///   _b = move? _a
///   _c = move? _a
///   _d = move? _c
/// where each of the locals is only assigned once.
///
/// We want to replace all those locals by `copy _a`.
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
        for (local, &head) in ssa.copy_classes().iter_enumerated() {
            if local != head {
                any_replacement = true;
                unified.insert(head);
                unified.insert(local);
            }
        }

        if !any_replacement {
            return;
        }

        Replacer { tcx, copy_classes: ssa.copy_classes(), unified }.visit_body_preserves_cfg(body);

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
    copy_classes: &'a IndexSlice<Local, Local>,
}

impl<'tcx> MutVisitor<'tcx> for Replacer<'_, 'tcx> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    #[tracing::instrument(level = "trace", skip(self))]
    fn visit_local(&mut self, local: &mut Local, ctxt: PlaceContext, _: Location) {
        *local = self.copy_classes[*local];
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
            && self.unified.contains(l)
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
