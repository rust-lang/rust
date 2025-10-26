use rustc_index::bit_set::DenseBitSet;
use rustc_middle::mir::visit::Visitor;
use rustc_middle::mir::*;

use crate::{Analysis, GenKill};

/// A dataflow analysis that tracks whether a pointer or reference could possibly exist that points
/// to a given local. This analysis ignores fake borrows, so it should not be used by
/// borrowck.
///
/// At present, this is used as a very limited form of alias analysis. For example,
/// `MaybeBorrowedLocals` is used to compute which locals are live during a yield expression for
/// immovable coroutines.
#[derive(Clone)]
pub struct MaybeBorrowedLocals;

impl MaybeBorrowedLocals {
    pub(super) fn transfer_function<'a, T>(trans: &'a mut T) -> TransferFunction<'a, T> {
        TransferFunction { trans }
    }
}

impl<'tcx> Analysis<'tcx> for MaybeBorrowedLocals {
    type Domain = DenseBitSet<Local>;
    const NAME: &'static str = "maybe_borrowed_locals";

    fn bottom_value(&self, body: &Body<'tcx>) -> Self::Domain {
        // bottom = unborrowed
        DenseBitSet::new_empty(body.local_decls().len())
    }

    fn initialize_start_block(&self, _: &Body<'tcx>, _: &mut Self::Domain) {
        // No locals are aliased on function entry
    }

    fn apply_primary_statement_effect(
        &mut self,
        state: &mut Self::Domain,
        statement: &Statement<'tcx>,
        location: Location,
    ) {
        Self::transfer_function(state).visit_statement(statement, location);
    }

    fn apply_primary_terminator_effect<'mir>(
        &mut self,
        state: &mut Self::Domain,
        terminator: &'mir Terminator<'tcx>,
        location: Location,
    ) -> TerminatorEdges<'mir, 'tcx> {
        Self::transfer_function(state).visit_terminator(terminator, location);
        terminator.edges()
    }
}

/// A `Visitor` that defines the transfer function for `MaybeBorrowedLocals`.
pub(super) struct TransferFunction<'a, T> {
    trans: &'a mut T,
}

impl<'tcx, T> Visitor<'tcx> for TransferFunction<'_, T>
where
    T: GenKill<Local>,
{
    fn visit_statement(&mut self, stmt: &Statement<'tcx>, location: Location) {
        self.super_statement(stmt, location);

        // When we reach a `StorageDead` statement, we can assume that any pointers to this memory
        // are now invalid.
        if let StatementKind::StorageDead(local) = stmt.kind {
            self.trans.kill(local);
        }
    }

    fn visit_rvalue(&mut self, rvalue: &Rvalue<'tcx>, location: Location) {
        self.super_rvalue(rvalue, location);

        match rvalue {
            // We ignore fake borrows as these get removed after analysis and shouldn't effect
            // the layout of generators.
            Rvalue::RawPtr(_, borrowed_place)
            | Rvalue::Ref(_, BorrowKind::Mut { .. } | BorrowKind::Shared, borrowed_place) => {
                if !borrowed_place.is_indirect() {
                    self.trans.gen_(borrowed_place.local);
                }
            }

            Rvalue::Cast(..)
            | Rvalue::Ref(_, BorrowKind::Fake(_), _)
            | Rvalue::ShallowInitBox(..)
            | Rvalue::Use(..)
            | Rvalue::ThreadLocalRef(..)
            | Rvalue::Repeat(..)
            | Rvalue::BinaryOp(..)
            | Rvalue::NullaryOp(..)
            | Rvalue::UnaryOp(..)
            | Rvalue::Discriminant(..)
            | Rvalue::Aggregate(..)
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
                if !dropped_place.is_indirect() {
                    self.trans.gen_(dropped_place.local);
                }
            }

            TerminatorKind::UnwindTerminate(_)
            | TerminatorKind::Assert { .. }
            | TerminatorKind::Call { .. }
            | TerminatorKind::FalseEdge { .. }
            | TerminatorKind::FalseUnwind { .. }
            | TerminatorKind::CoroutineDrop
            | TerminatorKind::Goto { .. }
            | TerminatorKind::InlineAsm { .. }
            | TerminatorKind::UnwindResume
            | TerminatorKind::Return
            | TerminatorKind::TailCall { .. }
            | TerminatorKind::SwitchInt { .. }
            | TerminatorKind::Unreachable
            | TerminatorKind::Yield { .. } => {}
        }
    }
}

/// The set of locals that are borrowed at some point in the MIR body.
pub fn borrowed_locals(body: &Body<'_>) -> DenseBitSet<Local> {
    struct Borrowed(DenseBitSet<Local>);

    impl GenKill<Local> for Borrowed {
        #[inline]
        fn gen_(&mut self, elem: Local) {
            self.0.gen_(elem)
        }
        #[inline]
        fn kill(&mut self, _: Local) {
            // Ignore borrow invalidation.
        }
    }

    let mut borrowed = Borrowed(DenseBitSet::new_empty(body.local_decls.len()));
    TransferFunction { trans: &mut borrowed }.visit_body(body);
    borrowed.0
}
