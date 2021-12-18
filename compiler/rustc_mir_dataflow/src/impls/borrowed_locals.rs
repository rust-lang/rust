use super::*;

use crate::{AnalysisDomain, CallReturnPlaces, GenKill, GenKillAnalysis};
use rustc_middle::mir::visit::Visitor;
use rustc_middle::mir::*;

/// A dataflow analysis that tracks whether a pointer or reference could possibly exist that points
/// to a given local.
///
/// At present, this is used as a very limited form of alias analysis. For example,
/// `MaybeBorrowedLocals` is used to compute which locals are live during a yield expression for
/// immovable generators.
pub struct MaybeBorrowedLocals {
    ignore_borrow_on_drop: bool,
}

impl MaybeBorrowedLocals {
    /// A dataflow analysis that records whether a pointer or reference exists that may alias the
    /// given local.
    pub fn all_borrows() -> Self {
        MaybeBorrowedLocals { ignore_borrow_on_drop: false }
    }
}

impl MaybeBorrowedLocals {
    /// During dataflow analysis, ignore the borrow that may occur when a place is dropped.
    ///
    /// Drop terminators may call custom drop glue (`Drop::drop`), which takes `&mut self` as a
    /// parameter. In the general case, a drop impl could launder that reference into the
    /// surrounding environment through a raw pointer, thus creating a valid `*mut` pointing to the
    /// dropped local. We are not yet willing to declare this particular case UB, so we must treat
    /// all dropped locals as mutably borrowed for now. See discussion on [#61069].
    ///
    /// In some contexts, we know that this borrow will never occur. For example, during
    /// const-eval, custom drop glue cannot be run. Code that calls this should document the
    /// assumptions that justify ignoring `Drop` terminators in this way.
    ///
    /// [#61069]: https://github.com/rust-lang/rust/pull/61069
    pub fn unsound_ignore_borrow_on_drop(self) -> Self {
        MaybeBorrowedLocals { ignore_borrow_on_drop: true, ..self }
    }

    fn transfer_function<'a, T>(&'a self, trans: &'a mut T) -> TransferFunction<'a, T> {
        TransferFunction { trans, ignore_borrow_on_drop: self.ignore_borrow_on_drop }
    }
}

impl<'tcx> AnalysisDomain<'tcx> for MaybeBorrowedLocals {
    type Domain = BitSet<Local>;
    const NAME: &'static str = "maybe_borrowed_locals";

    fn bottom_value(&self, body: &mir::Body<'tcx>) -> Self::Domain {
        // bottom = unborrowed
        BitSet::new_empty(body.local_decls().len())
    }

    fn initialize_start_block(&self, _: &mir::Body<'tcx>, _: &mut Self::Domain) {
        // No locals are aliased on function entry
    }
}

impl<'tcx> GenKillAnalysis<'tcx> for MaybeBorrowedLocals {
    type Idx = Local;

    fn statement_effect(
        &self,
        trans: &mut impl GenKill<Self::Idx>,
        statement: &mir::Statement<'tcx>,
        location: Location,
    ) {
        self.transfer_function(trans).visit_statement(statement, location);
    }

    fn terminator_effect(
        &self,
        trans: &mut impl GenKill<Self::Idx>,
        terminator: &mir::Terminator<'tcx>,
        location: Location,
    ) {
        self.transfer_function(trans).visit_terminator(terminator, location);
    }

    fn call_return_effect(
        &self,
        _trans: &mut impl GenKill<Self::Idx>,
        _block: mir::BasicBlock,
        _return_places: CallReturnPlaces<'_, 'tcx>,
    ) {
    }
}

/// A `Visitor` that defines the transfer function for `MaybeBorrowedLocals`.
struct TransferFunction<'a, T> {
    trans: &'a mut T,
    ignore_borrow_on_drop: bool,
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

    fn visit_rvalue(&mut self, rvalue: &mir::Rvalue<'tcx>, location: Location) {
        self.super_rvalue(rvalue, location);

        match rvalue {
            mir::Rvalue::AddressOf(_mt, borrowed_place) => {
                if !borrowed_place.is_indirect() {
                    self.trans.gen(borrowed_place.local);
                }
            }

            mir::Rvalue::Ref(_, _kind, borrowed_place) => {
                if !borrowed_place.is_indirect() {
                    self.trans.gen(borrowed_place.local);
                }
            }

            mir::Rvalue::Cast(..)
            | mir::Rvalue::ShallowInitBox(..)
            | mir::Rvalue::Use(..)
            | mir::Rvalue::ThreadLocalRef(..)
            | mir::Rvalue::Repeat(..)
            | mir::Rvalue::Len(..)
            | mir::Rvalue::BinaryOp(..)
            | mir::Rvalue::CheckedBinaryOp(..)
            | mir::Rvalue::NullaryOp(..)
            | mir::Rvalue::UnaryOp(..)
            | mir::Rvalue::Discriminant(..)
            | mir::Rvalue::Aggregate(..) => {}
        }
    }

    fn visit_terminator(&mut self, terminator: &mir::Terminator<'tcx>, location: Location) {
        self.super_terminator(terminator, location);

        match terminator.kind {
            mir::TerminatorKind::Drop { place: dropped_place, .. }
            | mir::TerminatorKind::DropAndReplace { place: dropped_place, .. } => {
                // See documentation for `unsound_ignore_borrow_on_drop` for an explanation.
                if !self.ignore_borrow_on_drop {
                    self.trans.gen(dropped_place.local);
                }
            }

            TerminatorKind::Abort
            | TerminatorKind::Assert { .. }
            | TerminatorKind::Call { .. }
            | TerminatorKind::FalseEdge { .. }
            | TerminatorKind::FalseUnwind { .. }
            | TerminatorKind::GeneratorDrop
            | TerminatorKind::Goto { .. }
            | TerminatorKind::InlineAsm { .. }
            | TerminatorKind::Resume
            | TerminatorKind::Return
            | TerminatorKind::SwitchInt { .. }
            | TerminatorKind::Unreachable
            | TerminatorKind::Yield { .. } => {}
        }
    }
}
