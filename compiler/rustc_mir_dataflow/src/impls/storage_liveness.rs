pub use super::*;

use crate::storage::AlwaysLiveLocals;
use crate::{CallReturnPlaces, GenKill, Results, ResultsRefCursor};
use rustc_middle::mir::visit::{NonMutatingUseContext, PlaceContext, Visitor};
use rustc_middle::mir::*;
use std::cell::RefCell;

#[derive(Clone)]
pub struct MaybeStorageLive {
    always_live_locals: AlwaysLiveLocals,
}

impl MaybeStorageLive {
    pub fn new(always_live_locals: AlwaysLiveLocals) -> Self {
        MaybeStorageLive { always_live_locals }
    }
}

impl<'tcx> crate::AnalysisDomain<'tcx> for MaybeStorageLive {
    type Domain = BitSet<Local>;

    const NAME: &'static str = "maybe_storage_live";

    fn bottom_value(&self, body: &mir::Body<'tcx>) -> Self::Domain {
        // bottom = dead
        BitSet::new_empty(body.local_decls.len())
    }

    fn initialize_start_block(&self, body: &mir::Body<'tcx>, on_entry: &mut Self::Domain) {
        assert_eq!(body.local_decls.len(), self.always_live_locals.domain_size());
        for local in self.always_live_locals.iter() {
            on_entry.insert(local);
        }

        for arg in body.args_iter() {
            on_entry.insert(arg);
        }
    }
}

impl<'tcx> crate::GenKillAnalysis<'tcx> for MaybeStorageLive {
    type Idx = Local;

    fn statement_effect(
        &self,
        trans: &mut impl GenKill<Self::Idx>,
        stmt: &mir::Statement<'tcx>,
        _: Location,
    ) {
        match stmt.kind {
            StatementKind::StorageLive(l) => trans.gen(l),
            StatementKind::StorageDead(l) => trans.kill(l),
            _ => (),
        }
    }

    fn terminator_effect(
        &self,
        _trans: &mut impl GenKill<Self::Idx>,
        _: &mir::Terminator<'tcx>,
        _: Location,
    ) {
        // Terminators have no effect
    }

    fn call_return_effect(
        &self,
        _trans: &mut impl GenKill<Self::Idx>,
        _block: BasicBlock,
        _return_places: CallReturnPlaces<'_, 'tcx>,
    ) {
        // Nothing to do when a call returns successfully
    }
}

type BorrowedLocalsResults<'a, 'tcx> = ResultsRefCursor<'a, 'a, 'tcx, MaybeBorrowedLocals>;

/// Dataflow analysis that determines whether each local requires storage at a
/// given location; i.e. whether its storage can go away without being observed.
pub struct MaybeRequiresStorage<'mir, 'tcx> {
    body: &'mir Body<'tcx>,
    borrowed_locals: RefCell<BorrowedLocalsResults<'mir, 'tcx>>,
}

impl<'mir, 'tcx> MaybeRequiresStorage<'mir, 'tcx> {
    pub fn new(
        body: &'mir Body<'tcx>,
        borrowed_locals: &'mir Results<'tcx, MaybeBorrowedLocals>,
    ) -> Self {
        MaybeRequiresStorage {
            body,
            borrowed_locals: RefCell::new(ResultsRefCursor::new(&body, borrowed_locals)),
        }
    }
}

impl<'mir, 'tcx> crate::AnalysisDomain<'tcx> for MaybeRequiresStorage<'mir, 'tcx> {
    type Domain = BitSet<Local>;

    const NAME: &'static str = "requires_storage";

    fn bottom_value(&self, body: &mir::Body<'tcx>) -> Self::Domain {
        // bottom = dead
        BitSet::new_empty(body.local_decls.len())
    }

    fn initialize_start_block(&self, body: &mir::Body<'tcx>, on_entry: &mut Self::Domain) {
        // The resume argument is live on function entry (we don't care about
        // the `self` argument)
        for arg in body.args_iter().skip(1) {
            on_entry.insert(arg);
        }
    }
}

impl<'mir, 'tcx> crate::GenKillAnalysis<'tcx> for MaybeRequiresStorage<'mir, 'tcx> {
    type Idx = Local;

    fn before_statement_effect(
        &self,
        trans: &mut impl GenKill<Self::Idx>,
        stmt: &mir::Statement<'tcx>,
        loc: Location,
    ) {
        // If a place is borrowed in a statement, it needs storage for that statement.
        self.borrowed_locals.borrow().analysis().statement_effect(trans, stmt, loc);

        match &stmt.kind {
            StatementKind::StorageDead(l) => trans.kill(*l),

            // If a place is assigned to in a statement, it needs storage for that statement.
            StatementKind::Assign(box (place, _))
            | StatementKind::SetDiscriminant { box place, .. } => {
                trans.gen(place.local);
            }

            // Nothing to do for these. Match exhaustively so this fails to compile when new
            // variants are added.
            StatementKind::AscribeUserType(..)
            | StatementKind::Coverage(..)
            | StatementKind::FakeRead(..)
            | StatementKind::Nop
            | StatementKind::Retag(..)
            | StatementKind::CopyNonOverlapping(..)
            | StatementKind::StorageLive(..) => {}
        }
    }

    fn statement_effect(
        &self,
        trans: &mut impl GenKill<Self::Idx>,
        _: &mir::Statement<'tcx>,
        loc: Location,
    ) {
        // If we move from a place then only stops needing storage *after*
        // that statement.
        self.check_for_move(trans, loc);
    }

    fn before_terminator_effect(
        &self,
        trans: &mut impl GenKill<Self::Idx>,
        terminator: &mir::Terminator<'tcx>,
        loc: Location,
    ) {
        // If a place is borrowed in a terminator, it needs storage for that terminator.
        self.borrowed_locals.borrow().analysis().terminator_effect(trans, terminator, loc);

        match &terminator.kind {
            TerminatorKind::Call { destination: Some((place, _)), .. } => {
                trans.gen(place.local);
            }

            // Note that we do *not* gen the `resume_arg` of `Yield` terminators. The reason for
            // that is that a `yield` will return from the function, and `resume_arg` is written
            // only when the generator is later resumed. Unlike `Call`, this doesn't require the
            // place to have storage *before* the yield, only after.
            TerminatorKind::Yield { .. } => {}

            TerminatorKind::InlineAsm { operands, .. } => {
                for op in operands {
                    match op {
                        InlineAsmOperand::Out { place, .. }
                        | InlineAsmOperand::InOut { out_place: place, .. } => {
                            if let Some(place) = place {
                                trans.gen(place.local);
                            }
                        }
                        InlineAsmOperand::In { .. }
                        | InlineAsmOperand::Const { .. }
                        | InlineAsmOperand::SymFn { .. }
                        | InlineAsmOperand::SymStatic { .. } => {}
                    }
                }
            }

            // Nothing to do for these. Match exhaustively so this fails to compile when new
            // variants are added.
            TerminatorKind::Call { destination: None, .. }
            | TerminatorKind::Abort
            | TerminatorKind::Assert { .. }
            | TerminatorKind::Drop { .. }
            | TerminatorKind::DropAndReplace { .. }
            | TerminatorKind::FalseEdge { .. }
            | TerminatorKind::FalseUnwind { .. }
            | TerminatorKind::GeneratorDrop
            | TerminatorKind::Goto { .. }
            | TerminatorKind::Resume
            | TerminatorKind::Return
            | TerminatorKind::SwitchInt { .. }
            | TerminatorKind::Unreachable => {}
        }
    }

    fn terminator_effect(
        &self,
        trans: &mut impl GenKill<Self::Idx>,
        terminator: &mir::Terminator<'tcx>,
        loc: Location,
    ) {
        match terminator.kind {
            // For call terminators the destination requires storage for the call
            // and after the call returns successfully, but not after a panic.
            // Since `propagate_call_unwind` doesn't exist, we have to kill the
            // destination here, and then gen it again in `call_return_effect`.
            TerminatorKind::Call { destination: Some((place, _)), .. } => {
                trans.kill(place.local);
            }

            // The same applies to InlineAsm outputs.
            TerminatorKind::InlineAsm { ref operands, .. } => {
                CallReturnPlaces::InlineAsm(operands).for_each(|place| trans.kill(place.local));
            }

            // Nothing to do for these. Match exhaustively so this fails to compile when new
            // variants are added.
            TerminatorKind::Call { destination: None, .. }
            | TerminatorKind::Yield { .. }
            | TerminatorKind::Abort
            | TerminatorKind::Assert { .. }
            | TerminatorKind::Drop { .. }
            | TerminatorKind::DropAndReplace { .. }
            | TerminatorKind::FalseEdge { .. }
            | TerminatorKind::FalseUnwind { .. }
            | TerminatorKind::GeneratorDrop
            | TerminatorKind::Goto { .. }
            | TerminatorKind::Resume
            | TerminatorKind::Return
            | TerminatorKind::SwitchInt { .. }
            | TerminatorKind::Unreachable => {}
        }

        self.check_for_move(trans, loc);
    }

    fn call_return_effect(
        &self,
        trans: &mut impl GenKill<Self::Idx>,
        _block: BasicBlock,
        return_places: CallReturnPlaces<'_, 'tcx>,
    ) {
        return_places.for_each(|place| trans.gen(place.local));
    }

    fn yield_resume_effect(
        &self,
        trans: &mut impl GenKill<Self::Idx>,
        _resume_block: BasicBlock,
        resume_place: mir::Place<'tcx>,
    ) {
        trans.gen(resume_place.local);
    }
}

impl<'mir, 'tcx> MaybeRequiresStorage<'mir, 'tcx> {
    /// Kill locals that are fully moved and have not been borrowed.
    fn check_for_move(&self, trans: &mut impl GenKill<Local>, loc: Location) {
        let mut visitor = MoveVisitor { trans, borrowed_locals: &self.borrowed_locals };
        visitor.visit_location(&self.body, loc);
    }
}

struct MoveVisitor<'a, 'mir, 'tcx, T> {
    borrowed_locals: &'a RefCell<BorrowedLocalsResults<'mir, 'tcx>>,
    trans: &'a mut T,
}

impl<'a, 'mir, 'tcx, T> Visitor<'tcx> for MoveVisitor<'a, 'mir, 'tcx, T>
where
    T: GenKill<Local>,
{
    fn visit_local(&mut self, local: &Local, context: PlaceContext, loc: Location) {
        if PlaceContext::NonMutatingUse(NonMutatingUseContext::Move) == context {
            let mut borrowed_locals = self.borrowed_locals.borrow_mut();
            borrowed_locals.seek_before_primary_effect(loc);
            if !borrowed_locals.contains(*local) {
                self.trans.kill(*local);
            }
        }
    }
}
