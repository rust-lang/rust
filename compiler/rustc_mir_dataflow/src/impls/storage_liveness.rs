pub use super::*;

use crate::impls::BorrowedLocalsResultsCursor;
use crate::{CallReturnPlaces, GenKill};
use rustc_middle::mir::visit::{NonMutatingUseContext, PlaceContext, Visitor};
use rustc_middle::mir::*;
use std::borrow::Cow;

#[derive(Clone)]
pub struct MaybeStorageLive<'a> {
    always_live_locals: Cow<'a, BitSet<Local>>,
}

impl<'a> MaybeStorageLive<'a> {
    pub fn new(always_live_locals: Cow<'a, BitSet<Local>>) -> Self {
        MaybeStorageLive { always_live_locals }
    }
}

impl crate::CloneAnalysis for MaybeStorageLive<'_> {
    fn clone_analysis(&self) -> Self {
        self.clone()
    }
}

impl<'tcx, 'a> crate::AnalysisDomain<'tcx> for MaybeStorageLive<'a> {
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

impl<'tcx, 'a> crate::GenKillAnalysis<'tcx> for MaybeStorageLive<'a> {
    type Idx = Local;

    fn statement_effect(
        &mut self,
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
        &mut self,
        _trans: &mut impl GenKill<Self::Idx>,
        _: &mir::Terminator<'tcx>,
        _: Location,
    ) {
        // Terminators have no effect
    }

    fn call_return_effect(
        &mut self,
        _trans: &mut impl GenKill<Self::Idx>,
        _block: BasicBlock,
        _return_places: CallReturnPlaces<'_, 'tcx>,
    ) {
        // Nothing to do when a call returns successfully
    }
}

#[derive(Clone)]
pub struct MaybeStorageDead {
    always_live_locals: BitSet<Local>,
}

impl MaybeStorageDead {
    pub fn new(always_live_locals: BitSet<Local>) -> Self {
        MaybeStorageDead { always_live_locals }
    }
}

impl<'tcx> crate::AnalysisDomain<'tcx> for MaybeStorageDead {
    type Domain = BitSet<Local>;

    const NAME: &'static str = "maybe_storage_dead";

    fn bottom_value(&self, body: &mir::Body<'tcx>) -> Self::Domain {
        // bottom = live
        BitSet::new_empty(body.local_decls.len())
    }

    fn initialize_start_block(&self, body: &mir::Body<'tcx>, on_entry: &mut Self::Domain) {
        assert_eq!(body.local_decls.len(), self.always_live_locals.domain_size());
        // Do not iterate on return place and args, as they are trivially always live.
        for local in body.vars_and_temps_iter() {
            if !self.always_live_locals.contains(local) {
                on_entry.insert(local);
            }
        }
    }
}

impl<'tcx> crate::GenKillAnalysis<'tcx> for MaybeStorageDead {
    type Idx = Local;

    fn statement_effect(
        &mut self,
        trans: &mut impl GenKill<Self::Idx>,
        stmt: &mir::Statement<'tcx>,
        _: Location,
    ) {
        match stmt.kind {
            StatementKind::StorageLive(l) => trans.kill(l),
            StatementKind::StorageDead(l) => trans.gen(l),
            _ => (),
        }
    }

    fn terminator_effect(
        &mut self,
        _trans: &mut impl GenKill<Self::Idx>,
        _: &mir::Terminator<'tcx>,
        _: Location,
    ) {
        // Terminators have no effect
    }

    fn call_return_effect(
        &mut self,
        _trans: &mut impl GenKill<Self::Idx>,
        _block: BasicBlock,
        _return_places: CallReturnPlaces<'_, 'tcx>,
    ) {
        // Nothing to do when a call returns successfully
    }
}

/// Dataflow analysis that determines whether each local requires storage at a
/// given location; i.e. whether its storage can go away without being observed.
pub struct MaybeRequiresStorage<'a, 'mir, 'tcx> {
    body: &'mir Body<'tcx>,
    borrowed_locals_cursor: BorrowedLocalsResultsCursor<'a, 'mir, 'tcx>,
}

impl<'a, 'mir, 'tcx> MaybeRequiresStorage<'a, 'mir, 'tcx> {
    pub fn new(
        body: &'mir Body<'tcx>,
        borrowed_locals_cursor: BorrowedLocalsResultsCursor<'a, 'mir, 'tcx>,
    ) -> Self {
        MaybeRequiresStorage { body, borrowed_locals_cursor }
    }
}

impl<'a, 'mir, 'tcx> crate::AnalysisDomain<'tcx> for MaybeRequiresStorage<'a, 'mir, 'tcx> {
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

impl<'a, 'mir, 'tcx> crate::GenKillAnalysis<'tcx> for MaybeRequiresStorage<'a, 'mir, 'tcx> {
    type Idx = Local;

    fn before_statement_effect(
        &mut self,
        trans: &mut impl GenKill<Self::Idx>,
        stmt: &mir::Statement<'tcx>,
        loc: Location,
    ) {
        // If a place is borrowed in a statement, it needs storage for that statement.
        let borrowed_locals_at_loc = self.borrowed_locals_cursor.get(loc);
        for i in 0..self.body.local_decls().len() {
            let local = Local::from_usize(i);
            if borrowed_locals_at_loc.contains(local) {
                debug!("{:?} is borrowed", local);
                trans.gen(local);
            }
        }

        match &stmt.kind {
            StatementKind::StorageDead(l) => trans.kill(*l),

            // If a place is assigned to in a statement, it needs storage for that statement.
            StatementKind::Assign(box (place, _))
            | StatementKind::SetDiscriminant { box place, .. }
            | StatementKind::Deinit(box place) => {
                trans.gen(place.local);
            }

            // Nothing to do for these. Match exhaustively so this fails to compile when new
            // variants are added.
            StatementKind::AscribeUserType(..)
            | StatementKind::PlaceMention(..)
            | StatementKind::Coverage(..)
            | StatementKind::FakeRead(..)
            | StatementKind::ConstEvalCounter
            | StatementKind::Nop
            | StatementKind::Retag(..)
            | StatementKind::Intrinsic(..)
            | StatementKind::StorageLive(..) => {}
        }
    }

    fn statement_effect(
        &mut self,
        trans: &mut impl GenKill<Self::Idx>,
        _: &mir::Statement<'tcx>,
        loc: Location,
    ) {
        // If we move from a place then it only stops needing storage *after*
        // that statement.
        self.check_for_move(trans, loc);
    }

    fn before_terminator_effect(
        &mut self,
        trans: &mut impl GenKill<Self::Idx>,
        terminator: &mir::Terminator<'tcx>,
        loc: Location,
    ) {
        // If a place is borrowed in a statement, it needs storage for that statement.
        let borrowed_locals_at_loc = self.borrowed_locals_cursor.get(loc);
        for i in 0..self.body.local_decls().len() {
            let local = Local::from_usize(i);
            if borrowed_locals_at_loc.contains(local) {
                trans.gen(local);
            }
        }

        match &terminator.kind {
            TerminatorKind::Call { destination, .. } => {
                trans.gen(destination.local);
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
            TerminatorKind::Terminate
            | TerminatorKind::Assert { .. }
            | TerminatorKind::Drop { .. }
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
        &mut self,
        trans: &mut impl GenKill<Self::Idx>,
        terminator: &mir::Terminator<'tcx>,
        loc: Location,
    ) {
        match terminator.kind {
            // For call terminators the destination requires storage for the call
            // and after the call returns successfully, but not after a panic.
            // Since `propagate_call_unwind` doesn't exist, we have to kill the
            // destination here, and then gen it again in `call_return_effect`.
            TerminatorKind::Call { destination, .. } => {
                trans.kill(destination.local);
            }

            // The same applies to InlineAsm outputs.
            TerminatorKind::InlineAsm { ref operands, .. } => {
                CallReturnPlaces::InlineAsm(operands).for_each(|place| trans.kill(place.local));
            }

            // Nothing to do for these. Match exhaustively so this fails to compile when new
            // variants are added.
            TerminatorKind::Yield { .. }
            | TerminatorKind::Terminate
            | TerminatorKind::Assert { .. }
            | TerminatorKind::Drop { .. }
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
        &mut self,
        trans: &mut impl GenKill<Self::Idx>,
        _block: BasicBlock,
        return_places: CallReturnPlaces<'_, 'tcx>,
    ) {
        return_places.for_each(|place| trans.gen(place.local));
    }

    fn yield_resume_effect(
        &mut self,
        trans: &mut impl GenKill<Self::Idx>,
        _resume_block: BasicBlock,
        resume_place: mir::Place<'tcx>,
    ) {
        trans.gen(resume_place.local);
    }
}

impl<'a, 'mir, 'tcx> MaybeRequiresStorage<'a, 'mir, 'tcx> {
    /// Kill locals that are fully moved and have not been borrowed.
    fn check_for_move(&mut self, trans: &mut impl GenKill<Local>, loc: Location) {
        let body = self.body;
        let mut visitor = MoveVisitor { trans, borrowed_locals: &mut self.borrowed_locals_cursor };
        visitor.visit_location(body, loc);
    }
}

struct MoveVisitor<'a, 'b, 'mir, 'tcx, T> {
    borrowed_locals: &'a mut BorrowedLocalsResultsCursor<'b, 'mir, 'tcx>,
    trans: &'a mut T,
}

impl<'a, 'b, 'mir, 'tcx, T> Visitor<'tcx> for MoveVisitor<'a, 'b, 'mir, 'tcx, T>
where
    T: GenKill<Local>,
{
    #[instrument(skip(self), level = "debug")]
    fn visit_local(&mut self, local: Local, context: PlaceContext, loc: Location) {
        if PlaceContext::NonMutatingUse(NonMutatingUseContext::Move) == context {
            let borrowed_locals = self.borrowed_locals.get(loc);
            debug!(?borrowed_locals);
            if !borrowed_locals.contains(local) {
                self.trans.kill(local);
            }
        }
    }
}
