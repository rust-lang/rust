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
pub struct MaybeBorrowedLocals;

impl MaybeBorrowedLocals {
    fn transfer_function<'a, T>(&'a self, trans: &'a mut T) -> TransferFunction<'a, T> {
        TransferFunction { trans }
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
            mir::Rvalue::AddressOf(_, borrowed_place) | mir::Rvalue::Ref(_, _, borrowed_place) => {
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
            | mir::Rvalue::Aggregate(..)
            | mir::Rvalue::CopyForDeref(..) => {}
        }
    }

    fn visit_terminator(&mut self, terminator: &mir::Terminator<'tcx>, location: Location) {
        self.super_terminator(terminator, location);

        match terminator.kind {
            mir::TerminatorKind::Drop { place: dropped_place, .. } => {
                // Drop terminators may call custom drop glue (`Drop::drop`), which takes `&mut
                // self` as a parameter. In the general case, a drop impl could launder that
                // reference into the surrounding environment through a raw pointer, thus creating
                // a valid `*mut` pointing to the dropped local. We are not yet willing to declare
                // this particular case UB, so we must treat all dropped locals as mutably borrowed
                // for now. See discussion on [#61069].
                //
                // [#61069]: https://github.com/rust-lang/rust/pull/61069
                if !dropped_place.is_indirect() {
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

/// The set of locals that are borrowed at some point in the MIR body.
pub fn borrowed_locals(body: &Body<'_>) -> BitSet<Local> {
    struct Borrowed(BitSet<Local>);

    impl GenKill<Local> for Borrowed {
        #[inline]
        fn gen(&mut self, elem: Local) {
            self.0.gen(elem)
        }
        #[inline]
        fn kill(&mut self, _: Local) {
            // Ignore borrow invalidation.
        }
    }

    let mut borrowed = Borrowed(BitSet::new_empty(body.local_decls.len()));
    TransferFunction { trans: &mut borrowed }.visit_body(body);
    borrowed.0
}
