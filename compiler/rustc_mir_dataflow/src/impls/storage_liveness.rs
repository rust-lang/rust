use rustc_index::bit_set::BitSet;
use rustc_middle::mir::visit::{NonMutatingUseContext, PlaceContext, Visitor};
use rustc_middle::{mir::*, ty};

use std::borrow::Cow;

use super::MaybeBorrowedLocals;
use crate::{GenKill, ResultsCursor};

#[derive(Clone)]
pub struct MaybeStorageLive<'a> {
    always_live_locals: Cow<'a, BitSet<Local>>,
}

impl<'a> MaybeStorageLive<'a> {
    pub fn new(always_live_locals: Cow<'a, BitSet<Local>>) -> Self {
        MaybeStorageLive { always_live_locals }
    }
}

impl<'tcx, 'a> crate::AnalysisDomain<'tcx> for MaybeStorageLive<'a> {
    type Domain = BitSet<Local>;

    const NAME: &'static str = "maybe_storage_live";

    fn bottom_value(&self, _: &Body<'tcx>) -> Self::Domain {
        // bottom = dead
        BitSet::new_empty(self.always_live_locals.domain_size())
    }

    fn initialize_start_block(&self, body: &Body<'tcx>, on_entry: &mut Self::Domain) {
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

    fn domain_size(&self, _body: &Body<'tcx>) -> usize {
        self.always_live_locals.domain_size()
    }

    fn statement_effect(
        &mut self,
        trans: &mut impl GenKill<Self::Idx>,
        stmt: &Statement<'tcx>,
        _: Location,
    ) {
        match stmt.kind {
            StatementKind::StorageLive(l) => trans.gen(l),
            StatementKind::StorageDead(l) => trans.kill(l),
            _ => (),
        }
    }

    fn terminator_effect<'mir>(
        &mut self,
        _trans: &mut Self::Domain,
        terminator: &'mir Terminator<'tcx>,
        _: Location,
    ) -> TerminatorEdges<'mir, 'tcx> {
        // Terminators have no effect
        terminator.edges()
    }

    fn call_return_effect(
        &mut self,
        _trans: &mut Self::Domain,
        _block: BasicBlock,
        _return_places: CallReturnPlaces<'_, 'tcx>,
    ) {
        // Nothing to do when a call returns successfully
    }
}

#[derive(Clone)]
pub struct MaybeStorageDead<'a> {
    always_live_locals: Cow<'a, BitSet<Local>>,
}

impl<'a> MaybeStorageDead<'a> {
    pub fn new(always_live_locals: Cow<'a, BitSet<Local>>) -> Self {
        MaybeStorageDead { always_live_locals }
    }
}

impl<'tcx, 'a> crate::AnalysisDomain<'tcx> for MaybeStorageDead<'a> {
    type Domain = BitSet<Local>;

    const NAME: &'static str = "maybe_storage_dead";

    fn bottom_value(&self, _body: &Body<'tcx>) -> Self::Domain {
        // bottom = live
        BitSet::new_empty(self.always_live_locals.domain_size())
    }

    fn initialize_start_block(&self, body: &Body<'tcx>, on_entry: &mut Self::Domain) {
        // Do not iterate on return place and args, as they are trivially always live.
        for local in body.vars_and_temps_iter() {
            if !self.always_live_locals.contains(local) {
                on_entry.insert(local);
            }
        }
    }
}

impl<'tcx, 'a> crate::GenKillAnalysis<'tcx> for MaybeStorageDead<'a> {
    type Idx = Local;

    fn domain_size(&self, _body: &Body<'tcx>) -> usize {
        self.always_live_locals.domain_size()
    }

    fn statement_effect(
        &mut self,
        trans: &mut impl GenKill<Self::Idx>,
        stmt: &Statement<'tcx>,
        _: Location,
    ) {
        match stmt.kind {
            StatementKind::StorageLive(l) => trans.kill(l),
            StatementKind::StorageDead(l) => trans.gen(l),
            _ => (),
        }
    }

    fn terminator_effect<'mir>(
        &mut self,
        _: &mut Self::Domain,
        terminator: &'mir Terminator<'tcx>,
        _: Location,
    ) -> TerminatorEdges<'mir, 'tcx> {
        // Terminators have no effect
        terminator.edges()
    }

    fn call_return_effect(
        &mut self,
        _trans: &mut Self::Domain,
        _block: BasicBlock,
        _return_places: CallReturnPlaces<'_, 'tcx>,
    ) {
        // Nothing to do when a call returns successfully
    }
}

type BorrowedLocalsResults<'mir, 'tcx> = ResultsCursor<'mir, 'tcx, MaybeBorrowedLocals>;

/// Dataflow analysis that determines whether each local requires storage at a
/// given location; i.e. whether its storage can go away without being observed.
pub struct MaybeRequiresStorage<'mir, 'tcx> {
    borrowed_locals: BorrowedLocalsResults<'mir, 'tcx>,
    upvars: Option<(Local, usize)>,
}

impl<'mir, 'tcx> MaybeRequiresStorage<'mir, 'tcx> {
    pub fn new(borrowed_locals: BorrowedLocalsResults<'mir, 'tcx>) -> Self {
        MaybeRequiresStorage { borrowed_locals, upvars: None }
    }
    pub fn new_with_upvar(
        borrowed_locals: BorrowedLocalsResults<'mir, 'tcx>,
        upvars: (Local, usize),
    ) -> Self {
        MaybeRequiresStorage { borrowed_locals, upvars: Some(upvars) }
    }
}

impl<'tcx> crate::AnalysisDomain<'tcx> for MaybeRequiresStorage<'_, 'tcx> {
    type Domain = BitSet<Local>;

    const NAME: &'static str = "requires_storage";

    fn bottom_value(&self, body: &Body<'tcx>) -> Self::Domain {
        // bottom = dead
        BitSet::new_empty(body.local_decls.len() + self.upvars.map_or(0, |(_, nr_upvar)| nr_upvar))
    }

    fn initialize_start_block(&self, body: &Body<'tcx>, on_entry: &mut Self::Domain) {
        // The resume argument is live on function entry (we don't care about
        // the `self` argument)
        for arg in body.args_iter().skip(1) {
            on_entry.insert(arg);
        }
        if let Some((upvar_start, nr_upvar)) = self.upvars {
            assert_eq!(body.local_decls.next_index(), upvar_start);
            // The upvars requires storage on entry, too.
            for idx in 0..nr_upvar {
                on_entry.insert(upvar_start + idx);
            }
        }
    }
}

impl<'tcx> crate::GenKillAnalysis<'tcx> for MaybeRequiresStorage<'_, 'tcx> {
    type Idx = Local;

    fn domain_size(&self, body: &Body<'tcx>) -> usize {
        body.local_decls.len() + self.upvars.map_or(0, |(_, nr_upvar)| nr_upvar)
    }

    fn before_statement_effect(
        &mut self,
        trans: &mut impl GenKill<Self::Idx>,
        stmt: &Statement<'tcx>,
        loc: Location,
    ) {
        // If a place is borrowed in a statement, it needs storage for that statement.
        self.borrowed_locals.mut_analysis().statement_effect(trans, stmt, loc);

        match &stmt.kind {
            StatementKind::StorageDead(l) => trans.kill(*l),

            // If a place is assigned to in a statement, it needs storage for that statement.
            StatementKind::Assign(box (place, _))
            | StatementKind::SetDiscriminant { box place, .. }
            | StatementKind::Deinit(box place) => {
                if place.local == ty::CAPTURE_STRUCT_LOCAL
                    && let Some((upvar_start, nr_upvar)) = self.upvars
                {
                    match **place.projection {
                        [ProjectionElem::Field(field, _), ..] if field.as_usize() < nr_upvar => {
                            trans.gen(upvar_start + field.as_usize())
                        }
                        _ => bug!("unexpected upvar access"),
                    }
                } else {
                    trans.gen(place.local)
                }
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
        _: &Statement<'tcx>,
        loc: Location,
    ) {
        // If we move from a place then it only stops needing storage *after*
        // that statement.
        self.check_for_move(trans, loc);
    }

    fn before_terminator_effect(
        &mut self,
        trans: &mut Self::Domain,
        terminator: &Terminator<'tcx>,
        loc: Location,
    ) {
        // If a place is borrowed in a terminator, it needs storage for that terminator.
        self.borrowed_locals
            .mut_analysis()
            .transfer_function(trans)
            .visit_terminator(terminator, loc);

        match &terminator.kind {
            TerminatorKind::Call { destination, .. } => {
                if destination.local == ty::CAPTURE_STRUCT_LOCAL
                    && let Some((upvar_start, nr_upvar)) = self.upvars
                {
                    match **destination.projection {
                        [ProjectionElem::Field(field, _), ..] if field.as_usize() < nr_upvar => {
                            trans.gen(upvar_start + field.as_usize())
                        }
                        _ => bug!("unexpected upvar access"),
                    }
                } else {
                    trans.gen(destination.local);
                }
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
                                if place.local == ty::CAPTURE_STRUCT_LOCAL
                                    && let Some((upvar_start, nr_upvar)) = self.upvars
                                {
                                    match **place.projection {
                                        [ProjectionElem::Field(field, _), ..]
                                            if field.as_usize() < nr_upvar =>
                                        {
                                            trans.gen(upvar_start + field.as_usize())
                                        }
                                        _ => bug!("unexpected upvar access"),
                                    }
                                } else {
                                    trans.gen(place.local)
                                }
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
            | TerminatorKind::SwitchInt { .. }
            | TerminatorKind::Unreachable => {}
        }
    }

    fn terminator_effect<'t>(
        &mut self,
        trans: &mut Self::Domain,
        terminator: &'t Terminator<'tcx>,
        loc: Location,
    ) -> TerminatorEdges<'t, 'tcx> {
        match terminator.kind {
            // For call terminators the destination requires storage for the call
            // and after the call returns successfully, but not after a panic.
            // Since `propagate_call_unwind` doesn't exist, we have to kill the
            // destination here, and then gen it again in `call_return_effect`.
            TerminatorKind::Call { destination, .. } => {
                if destination.local == ty::CAPTURE_STRUCT_LOCAL
                    && let Some((upvar_start, nr_upvar)) = self.upvars
                {
                    match **destination.projection {
                        [ProjectionElem::Field(field, _), ..] if field.as_usize() < nr_upvar => {
                            trans.kill(upvar_start + field.as_usize())
                        }
                        _ => bug!("unexpected upvar access"),
                    }
                } else {
                    trans.kill(destination.local);
                }
            }

            // The same applies to InlineAsm outputs.
            TerminatorKind::InlineAsm { ref operands, .. } => {
                CallReturnPlaces::InlineAsm(operands).for_each(|place| {
                    if place.local == ty::CAPTURE_STRUCT_LOCAL
                        && let Some((upvar_start, nr_upvar)) = self.upvars
                    {
                        match **place.projection {
                            [ProjectionElem::Field(field, _), ..]
                                if field.as_usize() < nr_upvar =>
                            {
                                trans.kill(upvar_start + field.as_usize())
                            }
                            _ => bug!("unexpected upvar access"),
                        }
                    } else {
                        trans.kill(place.local)
                    }
                });
            }

            TerminatorKind::Drop { place, .. } => {
                if place.local == ty::CAPTURE_STRUCT_LOCAL
                    && let Some((upvar_start, nr_upvar)) = self.upvars
                {
                    match **place.projection {
                        [] => {
                            for field in 0..nr_upvar {
                                trans.kill(upvar_start + field)
                            }
                        }
                        [ProjectionElem::Field(field, _), ..] if field.as_usize() < nr_upvar => {
                            trans.kill(upvar_start + field.as_usize())
                        }
                        _ => bug!("unexpected upvar access"),
                    }
                }
            }

            // Nothing to do for these. Match exhaustively so this fails to compile when new
            // variants are added.
            TerminatorKind::Yield { .. }
            | TerminatorKind::UnwindTerminate(_)
            | TerminatorKind::Assert { .. }
            | TerminatorKind::FalseEdge { .. }
            | TerminatorKind::FalseUnwind { .. }
            | TerminatorKind::CoroutineDrop
            | TerminatorKind::Goto { .. }
            | TerminatorKind::UnwindResume
            | TerminatorKind::Return
            | TerminatorKind::SwitchInt { .. }
            | TerminatorKind::Unreachable => {}
        }

        self.check_for_move(trans, loc);
        terminator.edges()
    }

    fn call_return_effect(
        &mut self,
        trans: &mut Self::Domain,
        _block: BasicBlock,
        return_places: CallReturnPlaces<'_, 'tcx>,
    ) {
        return_places.for_each(|place| {
            if place.local == ty::CAPTURE_STRUCT_LOCAL
                && let Some((upvar_start, nr_upvar)) = self.upvars
            {
                match **place.projection {
                    [ProjectionElem::Field(field, _), ..] if field.as_usize() < nr_upvar => {
                        trans.gen(upvar_start + field.as_usize())
                    }
                    _ => bug!("unexpected upvar access"),
                }
            } else {
                trans.gen(place.local)
            }
        });
    }
}

impl<'tcx> MaybeRequiresStorage<'_, 'tcx> {
    /// Kill locals that are fully moved and have not been borrowed.
    fn check_for_move(&mut self, trans: &mut impl GenKill<Local>, loc: Location) {
        let body = self.borrowed_locals.body();
        let mut visitor = MoveVisitor {
            trans,
            borrowed_locals: &mut self.borrowed_locals,
            upvar_start: self.upvars.map(|(upvar_start, _)| upvar_start),
        };
        visitor.visit_location(body, loc);
    }
}

struct MoveVisitor<'a, 'mir, 'tcx, T> {
    borrowed_locals: &'a mut BorrowedLocalsResults<'mir, 'tcx>,
    trans: &'a mut T,
    upvar_start: Option<Local>,
}

impl<'tcx, T> Visitor<'tcx> for MoveVisitor<'_, '_, 'tcx, T>
where
    T: GenKill<Local>,
{
    fn visit_local(&mut self, local: Local, context: PlaceContext, loc: Location) {
        if matches!(context, PlaceContext::NonMutatingUse(NonMutatingUseContext::Move)) {
            self.borrowed_locals.seek_before_primary_effect(loc);
            if !self.borrowed_locals.contains(local) {
                self.trans.kill(local);
            }
        }
    }
    fn visit_operand(&mut self, operand: &Operand<'tcx>, location: Location) {
        if let (
            Operand::Move(Place { local: ty::CAPTURE_STRUCT_LOCAL, projection }),
            Some(upvar_start),
        ) = (operand, self.upvar_start)
            && let [ProjectionElem::Field(field, _)] = ***projection
        {
            // QUESTION: what to do with subprojection?
            self.borrowed_locals.seek_before_primary_effect(location);
            let local = upvar_start + field.as_usize();
            if !self.borrowed_locals.contains(local) {
                self.trans.kill(local);
            }
        } else {
            self.super_operand(operand, location);
        }
    }
}
