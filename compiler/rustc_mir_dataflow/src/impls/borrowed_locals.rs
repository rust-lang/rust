use rustc_index::bit_set::BitSet;
use rustc_middle::mir::*;
use rustc_middle::{mir::visit::Visitor, ty};

use crate::{AnalysisDomain, GenKill, GenKillAnalysis};

/// A dataflow analysis that tracks whether a pointer or reference could possibly exist that points
/// to a given local. This analysis ignores fake borrows, so it should not be used by
/// borrowck.
///
/// At present, this is used as a very limited form of alias analysis. For example,
/// `MaybeBorrowedLocals` is used to compute which locals are live during a yield expression for
/// immovable coroutines.
#[derive(Clone, Copy)]
pub struct MaybeBorrowedLocals {
    upvar_start: Option<(Local, usize)>,
}

impl MaybeBorrowedLocals {
    /// `upvar_start` is to signal that upvars are treated as locals,
    /// and locals greater than this value refers to upvars accessed
    /// through the tuple `ty::CAPTURE_STRUCT_LOCAL`, aka. _1.
    pub fn new(upvar_start: Option<(Local, usize)>) -> Self {
        Self { upvar_start }
    }

    pub(super) fn transfer_function<'a, T>(&'a self, trans: &'a mut T) -> TransferFunction<'a, T> {
        TransferFunction { trans, upvar_start: self.upvar_start }
    }

    pub fn domain_size(&self, body: &Body<'_>) -> usize {
        if let Some((start, len)) = self.upvar_start {
            start.as_usize() + len
        } else {
            body.local_decls.len()
        }
    }
}

impl<'tcx> AnalysisDomain<'tcx> for MaybeBorrowedLocals {
    type Domain = BitSet<Local>;
    const NAME: &'static str = "maybe_borrowed_locals";

    fn bottom_value(&self, body: &Body<'tcx>) -> Self::Domain {
        // bottom = unborrowed
        BitSet::new_empty(self.domain_size(body))
    }

    fn initialize_start_block(&self, _: &Body<'tcx>, _: &mut Self::Domain) {
        // No locals are aliased on function entry
    }
}

impl<'tcx> GenKillAnalysis<'tcx> for MaybeBorrowedLocals {
    type Idx = Local;

    fn domain_size(&self, body: &Body<'tcx>) -> usize {
        self.domain_size(body)
    }

    fn statement_effect(
        &mut self,
        trans: &mut impl GenKill<Self::Idx>,
        statement: &Statement<'tcx>,
        location: Location,
    ) {
        self.transfer_function(trans).visit_statement(statement, location);
    }

    fn terminator_effect<'mir>(
        &mut self,
        trans: &mut Self::Domain,
        terminator: &'mir Terminator<'tcx>,
        location: Location,
    ) -> TerminatorEdges<'mir, 'tcx> {
        self.transfer_function(trans).visit_terminator(terminator, location);
        terminator.edges()
    }

    fn call_return_effect(
        &mut self,
        _trans: &mut Self::Domain,
        _block: BasicBlock,
        _return_places: CallReturnPlaces<'_, 'tcx>,
    ) {
    }
}

/// A `Visitor` that defines the transfer function for `MaybeBorrowedLocals`.
pub(super) struct TransferFunction<'a, T> {
    trans: &'a mut T,
    upvar_start: Option<(Local, usize)>,
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
            Rvalue::AddressOf(_, borrowed_place)
            | Rvalue::Ref(_, BorrowKind::Mut { .. } | BorrowKind::Shared, borrowed_place) => {
                if !borrowed_place.is_indirect() {
                    if borrowed_place.local == ty::CAPTURE_STRUCT_LOCAL
                        && let Some((upvar_start, nr_upvars)) = self.upvar_start
                    {
                        match **borrowed_place.projection {
                            [ProjectionElem::Field(field, _), ..]
                                if field.as_usize() < nr_upvars =>
                            {
                                self.trans.gen(upvar_start + field.as_usize())
                            }
                            _ => bug!("unexpected upvar access"),
                        }
                    } else {
                        self.trans.gen(borrowed_place.local);
                    }
                }
            }

            Rvalue::Cast(..)
            | Rvalue::Ref(_, BorrowKind::Fake, _)
            | Rvalue::ShallowInitBox(..)
            | Rvalue::Use(..)
            | Rvalue::ThreadLocalRef(..)
            | Rvalue::Repeat(..)
            | Rvalue::Len(..)
            | Rvalue::BinaryOp(..)
            | Rvalue::CheckedBinaryOp(..)
            | Rvalue::NullaryOp(..)
            | Rvalue::UnaryOp(..)
            | Rvalue::Discriminant(..)
            | Rvalue::Aggregate(..)
            | Rvalue::CopyForDeref(..) => {}
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
                    if dropped_place.local == ty::CAPTURE_STRUCT_LOCAL
                        && let Some((upvar_start, nr_upvars)) = self.upvar_start
                    {
                        match **dropped_place.projection {
                            [] => {
                                for field in 0..nr_upvars {
                                    self.trans.gen(upvar_start + field)
                                }
                                self.trans.gen(dropped_place.local)
                            }
                            [ProjectionElem::Field(field, _), ..]
                                if field.as_usize() < nr_upvars =>
                            {
                                self.trans.gen(upvar_start + field.as_usize())
                            }
                            _ => bug!("unexpected upvar access"),
                        }
                    } else {
                        self.trans.gen(dropped_place.local);
                    }
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
    TransferFunction { trans: &mut borrowed, upvar_start: None }.visit_body(body);
    borrowed.0
}
