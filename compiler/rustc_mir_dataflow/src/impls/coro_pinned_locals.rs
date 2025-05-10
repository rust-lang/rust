use rustc_index::bit_set::DenseBitSet;
use rustc_middle::mir::visit::Visitor;
use rustc_middle::mir::*;
use tracing::debug;

use crate::{Analysis, GenKill};

#[derive(Clone)]
pub struct CoroutinePinnedLocals(pub Local);

impl CoroutinePinnedLocals {
    fn transfer_function<'a>(&self, domain: &'a mut DenseBitSet<Local>) -> TransferFunction<'a> {
        TransferFunction { local: self.0, trans: domain }
    }
}

impl<'tcx> Analysis<'tcx> for CoroutinePinnedLocals {
    type Domain = DenseBitSet<Local>;
    const NAME: &'static str = "coro_pinned_locals";

    fn bottom_value(&self, body: &Body<'tcx>) -> Self::Domain {
        // bottom = unborrowed
        DenseBitSet::new_empty(body.local_decls().len())
    }

    fn initialize_start_block(&self, _: &Body<'tcx>, _: &mut Self::Domain) {
        // No locals are actively borrowing from other locals on function entry
    }

    fn apply_primary_statement_effect(
        &mut self,
        state: &mut Self::Domain,
        statement: &Statement<'tcx>,
        location: Location,
    ) {
        self.transfer_function(state).visit_statement(statement, location);
    }

    fn apply_primary_terminator_effect<'mir>(
        &mut self,
        state: &mut Self::Domain,
        terminator: &'mir Terminator<'tcx>,
        location: Location,
    ) -> TerminatorEdges<'mir, 'tcx> {
        self.transfer_function(state).visit_terminator(terminator, location);

        terminator.edges()
    }
}

/// A `Visitor` that defines the transfer function for `CoroutinePinnedLocals`.
pub(super) struct TransferFunction<'a> {
    local: Local,
    trans: &'a mut DenseBitSet<Local>,
}

impl<'tcx> Visitor<'tcx> for TransferFunction<'_> {
    fn visit_statement(&mut self, statement: &Statement<'tcx>, location: Location) {
        self.super_statement(statement, location);

        if let StatementKind::StorageDead(local) = statement.kind {
            debug!(for_ = ?self.local, KILL = ?local, ?statement, ?location);
            self.trans.kill(local);
        }
    }

    fn visit_assign(
        &mut self,
        assigned_place: &Place<'tcx>,
        rvalue: &Rvalue<'tcx>,
        location: Location,
    ) {
        self.super_assign(assigned_place, rvalue, location);

        match rvalue {
            Rvalue::Ref(_, BorrowKind::Mut { .. } | BorrowKind::Shared, place)
            | Rvalue::RawPtr(RawPtrKind::Const | RawPtrKind::Mut, place) => {
                if (!place.is_indirect() && place.local == self.local)
                    || self.trans.contains(place.local)
                {
                    if assigned_place.is_indirect() {
                        debug!(for_ = ?self.local, GEN_ptr_indirect = ?assigned_place, borrowed_place = ?place, ?rvalue, ?location);
                        self.trans.gen_(self.local);
                    } else {
                        debug!(for_ = ?self.local, GEN_ptr_direct = ?assigned_place, borrowed_place = ?place, ?rvalue, ?location);
                        self.trans.gen_(assigned_place.local);
                    }
                }
            }

            // fake pointers don't count
            Rvalue::Ref(_, BorrowKind::Fake(_), _)
            | Rvalue::RawPtr(RawPtrKind::FakeForPtrMetadata, _) => {}

            Rvalue::Use(..)
            | Rvalue::Repeat(..)
            | Rvalue::ThreadLocalRef(..)
            | Rvalue::Len(..)
            | Rvalue::Cast(..)
            | Rvalue::BinaryOp(..)
            | Rvalue::NullaryOp(..)
            | Rvalue::UnaryOp(..)
            | Rvalue::Discriminant(..)
            | Rvalue::Aggregate(..)
            | Rvalue::ShallowInitBox(..)
            | Rvalue::CopyForDeref(..)
            | Rvalue::WrapUnsafeBinder(..) => {}
        }
    }

    fn visit_terminator(&mut self, terminator: &Terminator<'tcx>, location: Location) {
        self.super_terminator(terminator, location);

        match terminator.kind {
            TerminatorKind::Drop { place: dropped_place, .. } => {
                // Drop terminators may call custom drop glue (`Drop::drop`), which takes `&mut
                // self` as a parameter. In the general case, a drop impl could launder that
                // reference into the surrounding environment through a raw pointer, thus creating
                // a valid `*mut` pointing to the dropped local. We are not yet willing to declare
                // this particular case UB, so we must treat all dropped locals as mutably borrowed
                // for now. See discussion on [#61069].
                //
                // [#61069]: https://github.com/rust-lang/rust/pull/61069
                if !dropped_place.is_indirect() && dropped_place.local == self.local {
                    debug!(for_ = ?self.local, GEN_drop = ?dropped_place, ?terminator, ?location);
                    self.trans.gen_(self.local);
                }
            }

            TerminatorKind::Goto { .. }
            | TerminatorKind::SwitchInt { .. }
            | TerminatorKind::UnwindResume
            | TerminatorKind::UnwindTerminate(_)
            | TerminatorKind::Return
            | TerminatorKind::Unreachable
            | TerminatorKind::Call { .. }
            | TerminatorKind::TailCall { .. }
            | TerminatorKind::Assert { .. }
            | TerminatorKind::Yield { .. }
            | TerminatorKind::CoroutineDrop
            | TerminatorKind::FalseEdge { .. }
            | TerminatorKind::FalseUnwind { .. }
            | TerminatorKind::InlineAsm { .. } => {}
        }
    }
}
