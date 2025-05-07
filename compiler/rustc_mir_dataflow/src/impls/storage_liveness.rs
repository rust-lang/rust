use std::borrow::Cow;

use rustc_index::bit_set::DenseBitSet;
use rustc_middle::mir::visit::{NonMutatingUseContext, PlaceContext, Visitor};
use rustc_middle::mir::*;

use super::MaybeBorrowedLocals;
use crate::{Analysis, GenKill, ResultsCursor};

/// The set of locals in a MIR body that do not have `StorageLive`/`StorageDead` annotations.
///
/// These locals have fixed storage for the duration of the body.
pub fn always_storage_live_locals(body: &Body<'_>) -> DenseBitSet<Local> {
    let mut always_live_locals = DenseBitSet::new_filled(body.local_decls.len());

    for block in &*body.basic_blocks {
        for statement in &block.statements {
            if let StatementKind::StorageLive(l) | StatementKind::StorageDead(l) = statement.kind {
                always_live_locals.remove(l);
            }
        }
    }

    always_live_locals
}

pub struct MaybeStorageLive<'a> {
    always_live_locals: Cow<'a, DenseBitSet<Local>>,
}

impl<'a> MaybeStorageLive<'a> {
    pub fn new(always_live_locals: Cow<'a, DenseBitSet<Local>>) -> Self {
        MaybeStorageLive { always_live_locals }
    }
}

impl<'a, 'tcx> Analysis<'tcx> for MaybeStorageLive<'a> {
    type Domain = DenseBitSet<Local>;

    const NAME: &'static str = "maybe_storage_live";

    fn bottom_value(&self, body: &Body<'tcx>) -> Self::Domain {
        // bottom = dead
        DenseBitSet::new_empty(body.local_decls.len())
    }

    fn initialize_start_block(&self, body: &Body<'tcx>, state: &mut Self::Domain) {
        state.union(&*self.always_live_locals);

        for arg in body.args_iter() {
            state.insert(arg);
        }
    }

    fn apply_primary_statement_effect(
        &mut self,
        state: &mut Self::Domain,
        stmt: &Statement<'tcx>,
        _: Location,
    ) {
        match stmt.kind {
            StatementKind::StorageLive(l) => state.gen_(l),
            StatementKind::StorageDead(l) => state.kill(l),
            _ => (),
        }
    }
}

pub struct MaybeStorageDead<'a> {
    always_live_locals: Cow<'a, DenseBitSet<Local>>,
}

impl<'a> MaybeStorageDead<'a> {
    pub fn new(always_live_locals: Cow<'a, DenseBitSet<Local>>) -> Self {
        MaybeStorageDead { always_live_locals }
    }
}

impl<'a, 'tcx> Analysis<'tcx> for MaybeStorageDead<'a> {
    type Domain = DenseBitSet<Local>;

    const NAME: &'static str = "maybe_storage_dead";

    fn bottom_value(&self, body: &Body<'tcx>) -> Self::Domain {
        // bottom = live
        DenseBitSet::new_empty(body.local_decls.len())
    }

    fn initialize_start_block(&self, body: &Body<'tcx>, state: &mut Self::Domain) {
        assert_eq!(body.local_decls.len(), self.always_live_locals.domain_size());
        // Do not iterate on return place and args, as they are trivially always live.
        for local in body.vars_and_temps_iter() {
            if !self.always_live_locals.contains(local) {
                state.insert(local);
            }
        }
    }

    fn apply_primary_statement_effect(
        &mut self,
        state: &mut Self::Domain,
        stmt: &Statement<'tcx>,
        _: Location,
    ) {
        match stmt.kind {
            StatementKind::StorageLive(l) => state.kill(l),
            StatementKind::StorageDead(l) => state.gen_(l),
            _ => (),
        }
    }
}

type BorrowedLocalsResults<'mir, 'tcx> = ResultsCursor<'mir, 'tcx, MaybeBorrowedLocals>;

/// Dataflow analysis that determines whether each local requires storage at a
/// given location; i.e. whether its storage can go away without being observed.
pub struct MaybeRequiresStorage<'mir, 'tcx> {
    borrowed_locals: BorrowedLocalsResults<'mir, 'tcx>,
}

impl<'mir, 'tcx> MaybeRequiresStorage<'mir, 'tcx> {
    pub fn new(borrowed_locals: BorrowedLocalsResults<'mir, 'tcx>) -> Self {
        MaybeRequiresStorage { borrowed_locals }
    }
}

impl<'tcx> Analysis<'tcx> for MaybeRequiresStorage<'_, 'tcx> {
    type Domain = DenseBitSet<Local>;

    const NAME: &'static str = "requires_storage";

    fn bottom_value(&self, body: &Body<'tcx>) -> Self::Domain {
        // bottom = dead
        DenseBitSet::new_empty(body.local_decls.len())
    }

    fn initialize_start_block(&self, body: &Body<'tcx>, state: &mut Self::Domain) {
        // The resume argument is live on function entry (we don't care about
        // the `self` argument)
        for arg in body.args_iter().skip(1) {
            state.insert(arg);
        }
    }

    fn apply_early_statement_effect(
        &mut self,
        state: &mut Self::Domain,
        stmt: &Statement<'tcx>,
        loc: Location,
    ) {
        // If a place is borrowed in a statement, it needs storage for that statement.
        MaybeBorrowedLocals::transfer_function(state).visit_statement(stmt, loc);

        match &stmt.kind {
            StatementKind::StorageDead(l) => state.kill(*l),

            // If a place is assigned to in a statement, it needs storage for that statement.
            StatementKind::Assign(box (place, _))
            | StatementKind::SetDiscriminant { box place, .. }
            | StatementKind::Deinit(box place) => {
                state.gen_(place.local);
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
            | StatementKind::BackwardIncompatibleDropHint { .. }
            | StatementKind::StorageLive(..) => {}
        }
    }

    fn apply_primary_statement_effect(
        &mut self,
        state: &mut Self::Domain,
        _: &Statement<'tcx>,
        loc: Location,
    ) {
        // If we move from a place then it only stops needing storage *after*
        // that statement.
        self.check_for_move(state, loc);
    }

    fn apply_early_terminator_effect(
        &mut self,
        state: &mut Self::Domain,
        terminator: &Terminator<'tcx>,
        loc: Location,
    ) {
        // If a place is borrowed in a terminator, it needs storage for that terminator.
        MaybeBorrowedLocals::transfer_function(state).visit_terminator(terminator, loc);

        match &terminator.kind {
            TerminatorKind::Call { destination, .. } => {
                state.gen_(destination.local);
            }

            // Note that we do *not* gen the `resume_arg` of `Yield` terminators. The reason for
            // that is that a `yield` will return from the function, and `resume_arg` is written
            // only when the coroutine is later resumed. Unlike `Call`, this doesn't require the
            // place to have storage *before* the yield, only after.
            TerminatorKind::Yield { .. } => {}

            TerminatorKind::InlineAsm { operands, .. } => {
                for op in operands {
                    match op {
                        InlineAsmOperand::Out { place, .. }
                        | InlineAsmOperand::InOut { out_place: place, .. } => {
                            if let Some(place) = place {
                                state.gen_(place.local);
                            }
                        }
                        InlineAsmOperand::In { .. }
                        | InlineAsmOperand::Const { .. }
                        | InlineAsmOperand::SymFn { .. }
                        | InlineAsmOperand::SymStatic { .. }
                        | InlineAsmOperand::Label { .. } => {}
                    }
                }
            }

            // Nothing to do for these. Match exhaustively so this fails to compile when new
            // variants are added.
            TerminatorKind::UnwindTerminate(_)
            | TerminatorKind::Assert { .. }
            | TerminatorKind::Drop { .. }
            | TerminatorKind::FalseEdge { .. }
            | TerminatorKind::FalseUnwind { .. }
            | TerminatorKind::CoroutineDrop
            | TerminatorKind::Goto { .. }
            | TerminatorKind::UnwindResume
            | TerminatorKind::Return
            | TerminatorKind::TailCall { .. }
            | TerminatorKind::SwitchInt { .. }
            | TerminatorKind::Unreachable => {}
        }
    }

    fn apply_primary_terminator_effect<'t>(
        &mut self,
        state: &mut Self::Domain,
        terminator: &'t Terminator<'tcx>,
        loc: Location,
    ) -> TerminatorEdges<'t, 'tcx> {
        match terminator.kind {
            // For call terminators the destination requires storage for the call
            // and after the call returns successfully, but not after a panic.
            // Since `propagate_call_unwind` doesn't exist, we have to kill the
            // destination here, and then gen it again in `call_return_effect`.
            TerminatorKind::Call { destination, .. } => {
                state.kill(destination.local);
            }

            // The same applies to InlineAsm outputs.
            TerminatorKind::InlineAsm { ref operands, .. } => {
                CallReturnPlaces::InlineAsm(operands).for_each(|place| state.kill(place.local));
            }

            // Nothing to do for these. Match exhaustively so this fails to compile when new
            // variants are added.
            TerminatorKind::Yield { .. }
            | TerminatorKind::UnwindTerminate(_)
            | TerminatorKind::Assert { .. }
            | TerminatorKind::Drop { .. }
            | TerminatorKind::FalseEdge { .. }
            | TerminatorKind::FalseUnwind { .. }
            | TerminatorKind::CoroutineDrop
            | TerminatorKind::Goto { .. }
            | TerminatorKind::UnwindResume
            | TerminatorKind::Return
            | TerminatorKind::TailCall { .. }
            | TerminatorKind::SwitchInt { .. }
            | TerminatorKind::Unreachable => {}
        }

        self.check_for_move(state, loc);
        terminator.edges()
    }

    fn apply_call_return_effect(
        &mut self,
        state: &mut Self::Domain,
        _block: BasicBlock,
        return_places: CallReturnPlaces<'_, 'tcx>,
    ) {
        return_places.for_each(|place| state.gen_(place.local));
    }
}

impl<'tcx> MaybeRequiresStorage<'_, 'tcx> {
    /// Kill locals that are fully moved and have not been borrowed.
    fn check_for_move(&mut self, state: &mut <Self as Analysis<'tcx>>::Domain, loc: Location) {
        let body = self.borrowed_locals.body();
        let mut visitor = MoveVisitor { state, borrowed_locals: &mut self.borrowed_locals };
        visitor.visit_location(body, loc);
    }
}

struct MoveVisitor<'a, 'mir, 'tcx> {
    borrowed_locals: &'a mut BorrowedLocalsResults<'mir, 'tcx>,
    state: &'a mut DenseBitSet<Local>,
}

impl<'tcx> Visitor<'tcx> for MoveVisitor<'_, '_, 'tcx> {
    fn visit_local(&mut self, local: Local, context: PlaceContext, loc: Location) {
        if PlaceContext::NonMutatingUse(NonMutatingUseContext::Move) == context {
            self.borrowed_locals.seek_before_primary_effect(loc);
            if !self.borrowed_locals.get().contains(local) {
                self.state.kill(local);
            }
        }
    }
}
