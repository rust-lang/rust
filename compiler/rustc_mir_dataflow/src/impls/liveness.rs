use rustc_index::bit_set::BitSet;
use rustc_middle::mir::visit::{MutatingUseContext, NonMutatingUseContext, PlaceContext, Visitor};
use rustc_middle::mir::{
    self, CallReturnPlaces, Local, Location, Place, StatementKind, TerminatorEdges,
};

use crate::{Analysis, Backward, GenKill};

/// A [live-variable dataflow analysis][liveness].
///
/// This analysis considers references as being used only at the point of the
/// borrow. In other words, this analysis does not track uses because of references that already
/// exist. See [this `mir-dataflow` test][flow-test] for an example. You almost never want to use
/// this analysis without also looking at the results of [`MaybeBorrowedLocals`].
///
/// ## Field-(in)sensitivity
///
/// As the name suggests, this analysis is field insensitive. If a projection of a variable `x` is
/// assigned to (e.g. `x.0 = 42`), it does not "define" `x` as far as liveness is concerned. In fact,
/// such an assignment is currently marked as a "use" of `x` in an attempt to be maximally
/// conservative.
///
/// [`MaybeBorrowedLocals`]: super::MaybeBorrowedLocals
/// [flow-test]: https://github.com/rust-lang/rust/blob/a08c47310c7d49cbdc5d7afb38408ba519967ecd/src/test/ui/mir-dataflow/liveness-ptr.rs
/// [liveness]: https://en.wikipedia.org/wiki/Live_variable_analysis
pub struct MaybeLiveLocals;

impl<'tcx> Analysis<'tcx> for MaybeLiveLocals {
    type Domain = BitSet<Local>;
    type Direction = Backward;

    const NAME: &'static str = "liveness";

    fn bottom_value(&self, body: &mir::Body<'tcx>) -> Self::Domain {
        // bottom = not live
        BitSet::new_empty(body.local_decls.len())
    }

    fn initialize_start_block(&self, _: &mir::Body<'tcx>, _: &mut Self::Domain) {
        // No variables are live until we observe a use
    }

    fn apply_statement_effect(
        &mut self,
        trans: &mut Self::Domain,
        statement: &mir::Statement<'tcx>,
        location: Location,
    ) {
        TransferFunction(trans).visit_statement(statement, location);
    }

    fn apply_terminator_effect<'mir>(
        &mut self,
        trans: &mut Self::Domain,
        terminator: &'mir mir::Terminator<'tcx>,
        location: Location,
    ) -> TerminatorEdges<'mir, 'tcx> {
        TransferFunction(trans).visit_terminator(terminator, location);
        terminator.edges()
    }

    fn apply_call_return_effect(
        &mut self,
        trans: &mut Self::Domain,
        _block: mir::BasicBlock,
        return_places: CallReturnPlaces<'_, 'tcx>,
    ) {
        if let CallReturnPlaces::Yield(resume_place) = return_places {
            YieldResumeEffect(trans).visit_place(
                &resume_place,
                PlaceContext::MutatingUse(MutatingUseContext::Yield),
                Location::START,
            )
        } else {
            return_places.for_each(|place| {
                if let Some(local) = place.as_local() {
                    trans.kill(local);
                }
            });
        }
    }
}

pub struct TransferFunction<'a>(pub &'a mut BitSet<Local>);

impl<'tcx> Visitor<'tcx> for TransferFunction<'_> {
    fn visit_place(&mut self, place: &mir::Place<'tcx>, context: PlaceContext, location: Location) {
        if let PlaceContext::MutatingUse(MutatingUseContext::Yield) = context {
            // The resume place is evaluated and assigned to only after coroutine resumes, so its
            // effect is handled separately in `call_resume_effect`.
            return;
        }

        match DefUse::for_place(*place, context) {
            Some(DefUse::Def) => {
                if let PlaceContext::MutatingUse(
                    MutatingUseContext::Call | MutatingUseContext::AsmOutput,
                ) = context
                {
                    // For the associated terminators, this is only a `Def` when the terminator
                    // returns "successfully." As such, we handle this case separately in
                    // `call_return_effect` above. However, if the place looks like `*_5`, this is
                    // still unconditionally a use of `_5`.
                } else {
                    self.0.kill(place.local);
                }
            }
            Some(DefUse::Use) => self.0.gen_(place.local),
            None => {}
        }

        self.visit_projection(place.as_ref(), context, location);
    }

    fn visit_local(&mut self, local: Local, context: PlaceContext, _: Location) {
        DefUse::apply(self.0, local.into(), context);
    }
}

struct YieldResumeEffect<'a>(&'a mut BitSet<Local>);

impl<'tcx> Visitor<'tcx> for YieldResumeEffect<'_> {
    fn visit_place(&mut self, place: &mir::Place<'tcx>, context: PlaceContext, location: Location) {
        DefUse::apply(self.0, *place, context);
        self.visit_projection(place.as_ref(), context, location);
    }

    fn visit_local(&mut self, local: Local, context: PlaceContext, _: Location) {
        DefUse::apply(self.0, local.into(), context);
    }
}

#[derive(Eq, PartialEq, Clone)]
enum DefUse {
    Def,
    Use,
}

impl DefUse {
    fn apply(trans: &mut BitSet<Local>, place: Place<'_>, context: PlaceContext) {
        match DefUse::for_place(place, context) {
            Some(DefUse::Def) => trans.kill(place.local),
            Some(DefUse::Use) => trans.gen_(place.local),
            None => {}
        }
    }

    fn for_place(place: Place<'_>, context: PlaceContext) -> Option<DefUse> {
        match context {
            PlaceContext::NonUse(_) => None,

            PlaceContext::MutatingUse(
                MutatingUseContext::Call
                | MutatingUseContext::Yield
                | MutatingUseContext::AsmOutput
                | MutatingUseContext::Store
                | MutatingUseContext::Deinit,
            ) => {
                if place.is_indirect() {
                    // Treat derefs as a use of the base local. `*p = 4` is not a def of `p` but a
                    // use.
                    Some(DefUse::Use)
                } else if place.projection.is_empty() {
                    Some(DefUse::Def)
                } else {
                    None
                }
            }

            // Setting the discriminant is not a use because it does no reading, but it is also not
            // a def because it does not overwrite the whole place
            PlaceContext::MutatingUse(MutatingUseContext::SetDiscriminant) => {
                place.is_indirect().then_some(DefUse::Use)
            }

            // All other contexts are uses...
            PlaceContext::MutatingUse(
                MutatingUseContext::RawBorrow
                | MutatingUseContext::Borrow
                | MutatingUseContext::Drop
                | MutatingUseContext::Retag,
            )
            | PlaceContext::NonMutatingUse(
                NonMutatingUseContext::RawBorrow
                | NonMutatingUseContext::Copy
                | NonMutatingUseContext::Inspect
                | NonMutatingUseContext::Move
                | NonMutatingUseContext::PlaceMention
                | NonMutatingUseContext::FakeBorrow
                | NonMutatingUseContext::SharedBorrow,
            ) => Some(DefUse::Use),

            PlaceContext::MutatingUse(MutatingUseContext::Projection)
            | PlaceContext::NonMutatingUse(NonMutatingUseContext::Projection) => {
                unreachable!("A projection could be a def or a use and must be handled separately")
            }
        }
    }
}

/// Like `MaybeLiveLocals`, but does not mark locals as live if they are used in a dead assignment.
///
/// This is basically written for dead store elimination and nothing else.
///
/// All of the caveats of `MaybeLiveLocals` apply.
pub struct MaybeTransitiveLiveLocals<'a> {
    always_live: &'a BitSet<Local>,
}

impl<'a> MaybeTransitiveLiveLocals<'a> {
    /// The `always_alive` set is the set of locals to which all stores should unconditionally be
    /// considered live.
    ///
    /// This should include at least all locals that are ever borrowed.
    pub fn new(always_live: &'a BitSet<Local>) -> Self {
        MaybeTransitiveLiveLocals { always_live }
    }
}

impl<'a, 'tcx> Analysis<'tcx> for MaybeTransitiveLiveLocals<'a> {
    type Domain = BitSet<Local>;
    type Direction = Backward;

    const NAME: &'static str = "transitive liveness";

    fn bottom_value(&self, body: &mir::Body<'tcx>) -> Self::Domain {
        // bottom = not live
        BitSet::new_empty(body.local_decls.len())
    }

    fn initialize_start_block(&self, _: &mir::Body<'tcx>, _: &mut Self::Domain) {
        // No variables are live until we observe a use
    }

    fn apply_statement_effect(
        &mut self,
        trans: &mut Self::Domain,
        statement: &mir::Statement<'tcx>,
        location: Location,
    ) {
        // Compute the place that we are storing to, if any
        let destination = match &statement.kind {
            StatementKind::Assign(assign) => assign.1.is_safe_to_remove().then_some(assign.0),
            StatementKind::SetDiscriminant { place, .. } | StatementKind::Deinit(place) => {
                Some(**place)
            }
            StatementKind::FakeRead(_)
            | StatementKind::StorageLive(_)
            | StatementKind::StorageDead(_)
            | StatementKind::Retag(..)
            | StatementKind::AscribeUserType(..)
            | StatementKind::PlaceMention(..)
            | StatementKind::Coverage(..)
            | StatementKind::Intrinsic(..)
            | StatementKind::ConstEvalCounter
            | StatementKind::BackwardIncompatibleDropHint { .. }
            | StatementKind::Nop => None,
        };
        if let Some(destination) = destination {
            if !destination.is_indirect()
                && !trans.contains(destination.local)
                && !self.always_live.contains(destination.local)
            {
                // This store is dead
                return;
            }
        }
        TransferFunction(trans).visit_statement(statement, location);
    }

    fn apply_terminator_effect<'mir>(
        &mut self,
        trans: &mut Self::Domain,
        terminator: &'mir mir::Terminator<'tcx>,
        location: Location,
    ) -> TerminatorEdges<'mir, 'tcx> {
        TransferFunction(trans).visit_terminator(terminator, location);
        terminator.edges()
    }

    fn apply_call_return_effect(
        &mut self,
        trans: &mut Self::Domain,
        _block: mir::BasicBlock,
        return_places: CallReturnPlaces<'_, 'tcx>,
    ) {
        if let CallReturnPlaces::Yield(resume_place) = return_places {
            YieldResumeEffect(trans).visit_place(
                &resume_place,
                PlaceContext::MutatingUse(MutatingUseContext::Yield),
                Location::START,
            )
        } else {
            return_places.for_each(|place| {
                if let Some(local) = place.as_local() {
                    trans.remove(local);
                }
            });
        }
    }
}
