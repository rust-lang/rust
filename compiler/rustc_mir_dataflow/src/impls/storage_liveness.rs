use std::borrow::Cow;

use rustc_data_structures::fx::FxHashMap;
use rustc_index::bit_set::DenseBitSet;
use rustc_middle::mir::visit::{NonMutatingUseContext, PlaceContext, Visitor};
use rustc_middle::mir::*;
use smallvec::SmallVec;

use super::MaybeBorrowedLocals;
use crate::{Analysis, GenKill, Results, ResultsCursor};

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
        &self,
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
        &self,
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

/// For each location, records which locals can be killed by `MaybeRequiresStorage`.
type KillableLocals = FxHashMap<Location, SmallVec<[Local; 4]>>;

/// Dataflow analysis that determines whether each local requires storage at a
/// given location; i.e. whether its storage can go away without being observed.
pub struct MaybeRequiresStorage {
    /// Used to kill locals that are fully moved and have not been borrowed.
    killable_locals: KillableLocals,
}

impl MaybeRequiresStorage {
    pub fn new<'tcx>(
        body: &Body<'tcx>,
        borrowed_locals: &Results<'tcx, MaybeBorrowedLocals>,
    ) -> Self {
        struct KillableLocalsVisitor<'mir, 'tcx> {
            borrowed_locals_cursor: ResultsCursor<'mir, 'tcx, MaybeBorrowedLocals>,
            killable_locals: KillableLocals,
        }

        impl<'tcx> Visitor<'tcx> for KillableLocalsVisitor<'_, 'tcx> {
            fn visit_local(&mut self, local: Local, context: PlaceContext, loc: Location) {
                if PlaceContext::NonMutatingUse(NonMutatingUseContext::Move) == context {
                    self.borrowed_locals_cursor.seek_before_primary_effect(loc);
                    if !self.borrowed_locals_cursor.get().contains(local) {
                        self.killable_locals.entry(loc).or_default().push(local);
                    }
                }
            }
        }

        let mut visitor = KillableLocalsVisitor {
            borrowed_locals_cursor: ResultsCursor::new_borrowing(body, borrowed_locals),
            killable_locals: Default::default(),
        };

        for (bb, data) in traversal::reachable(body) {
            visitor.visit_basic_block_data(bb, data);
        }

        MaybeRequiresStorage { killable_locals: visitor.killable_locals }
    }
}

impl<'tcx> Analysis<'tcx> for MaybeRequiresStorage {
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
        &self,
        state: &mut Self::Domain,
        stmt: &Statement<'tcx>,
        _loc: Location,
    ) {
        // If a place is borrowed in a statement, it needs storage for that statement.
        MaybeBorrowedLocals::gen_statement(state, stmt);

        match &stmt.kind {
            StatementKind::StorageDead(l) => state.kill(*l),

            StatementKind::Assign((place, _)) => {
                state.gen_(place.local);
            }
            StatementKind::SetDiscriminant { place, .. } => {
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
            | StatementKind::Intrinsic(..)
            | StatementKind::BackwardIncompatibleDropHint { .. }
            | StatementKind::StorageLive(..) => {}
        }
    }

    fn apply_primary_statement_effect(
        &self,
        state: &mut Self::Domain,
        stmt: &Statement<'tcx>,
        loc: Location,
    ) {
        // If we move from a place then it only stops needing storage *after* that statement.
        self.check_for_move(state, loc);

        match &stmt.kind {
            // If a place is assigned to in a statement, it needs storage after that statement.
            // Even if the place was moved from in the rvalue (e.g. `x = x + 1` or `x = f(move x)`),
            // the assignment restores a valid value into the place.
            StatementKind::Assign((place, _)) => {
                state.gen_(place.local);
            }
            StatementKind::SetDiscriminant { place, .. } => {
                state.gen_(place.local);
            }

            StatementKind::StorageDead(_)
            | StatementKind::AscribeUserType(..)
            | StatementKind::PlaceMention(..)
            | StatementKind::Coverage(..)
            | StatementKind::FakeRead(..)
            | StatementKind::ConstEvalCounter
            | StatementKind::Nop
            | StatementKind::Intrinsic(..)
            | StatementKind::BackwardIncompatibleDropHint { .. }
            | StatementKind::StorageLive(..) => {}
        }
    }

    fn apply_early_terminator_effect(
        &self,
        state: &mut Self::Domain,
        terminator: &Terminator<'tcx>,
        _loc: Location,
    ) {
        // If a place is borrowed in a terminator, it needs storage for that terminator.
        MaybeBorrowedLocals::gen_terminator(state, terminator);

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

    fn apply_primary_terminator_effect(
        &self,
        state: &mut Self::Domain,
        terminator: &Terminator<'tcx>,
        loc: Location,
    ) {
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
    }

    fn apply_call_return_effect(
        &self,
        state: &mut Self::Domain,
        _block: BasicBlock,
        return_places: CallReturnPlaces<'_, 'tcx>,
    ) {
        return_places.for_each(|place| state.gen_(place.local));
    }
}

impl MaybeRequiresStorage {
    /// Kill locals that are fully moved and have not been borrowed.
    fn check_for_move(&self, state: &mut <Self as Analysis<'_>>::Domain, loc: Location) {
        if let Some(locals) = self.killable_locals.get(&loc) {
            for &l in locals {
                state.kill(l);
            }
        }
    }
}
