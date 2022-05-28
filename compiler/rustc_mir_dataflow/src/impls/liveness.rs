use rustc_index::bit_set::{BitSet, ChunkedBitSet};
use rustc_middle::mir::visit::{MutatingUseContext, NonMutatingUseContext, PlaceContext, Visitor};
use rustc_middle::mir::{self, Local, LocalDecls, Location, Place, StatementKind};
use rustc_middle::ty::TyCtxt;

use crate::{Analysis, AnalysisDomain, Backward, CallReturnPlaces, GenKill, GenKillAnalysis};

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

impl MaybeLiveLocals {
    fn transfer_function<'a, T>(&self, trans: &'a mut T) -> TransferFunction<'a, T> {
        TransferFunction(trans)
    }
}

impl<'tcx> AnalysisDomain<'tcx> for MaybeLiveLocals {
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
}

impl<'tcx> GenKillAnalysis<'tcx> for MaybeLiveLocals {
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
        trans: &mut impl GenKill<Self::Idx>,
        _block: mir::BasicBlock,
        return_places: CallReturnPlaces<'_, 'tcx>,
    ) {
        return_places.for_each(|place| {
            if let Some(local) = place.as_local() {
                trans.kill(local);
            }
        });
    }

    fn yield_resume_effect(
        &self,
        trans: &mut impl GenKill<Self::Idx>,
        _resume_block: mir::BasicBlock,
        resume_place: mir::Place<'tcx>,
    ) {
        if let Some(local) = resume_place.as_local() {
            trans.kill(local);
        }
    }
}

struct TransferFunction<'a, T>(&'a mut T);

impl<'tcx, T> Visitor<'tcx> for TransferFunction<'_, T>
where
    T: GenKill<Local>,
{
    fn visit_place(&mut self, place: &mir::Place<'tcx>, context: PlaceContext, location: Location) {
        let local = place.local;

        // We purposefully do not call `super_place` here to avoid calling `visit_local` for this
        // place with one of the `Projection` variants of `PlaceContext`.
        self.visit_projection(place.as_ref(), context, location);

        match DefUse::for_place(*place, context) {
            Some(DefUse::Def) => self.0.kill(local),
            Some(DefUse::Use) => self.0.gen(local),
            None => {}
        }
    }

    fn visit_local(&mut self, &local: &Local, context: PlaceContext, _: Location) {
        // Because we do not call `super_place` above, `visit_local` is only called for locals that
        // do not appear as part of  a `Place` in the MIR. This handles cases like the implicit use
        // of the return place in a `Return` terminator or the index in an `Index` projection.
        match DefUse::for_place(local.into(), context) {
            Some(DefUse::Def) => self.0.kill(local),
            Some(DefUse::Use) => self.0.gen(local),
            None => {}
        }
    }
}

#[derive(Eq, PartialEq, Clone)]
enum DefUse {
    Def,
    Use,
}

impl DefUse {
    fn for_place<'tcx>(place: Place<'tcx>, context: PlaceContext) -> Option<DefUse> {
        match context {
            PlaceContext::NonUse(_) => None,

            PlaceContext::MutatingUse(MutatingUseContext::Store | MutatingUseContext::Deinit) => {
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

            // For the associated terminators, this is only a `Def` when the terminator returns
            // "successfully." As such, we handle this case separately in `call_return_effect`
            // above. However, if the place looks like `*_5`, this is still unconditionally a use of
            // `_5`.
            PlaceContext::MutatingUse(
                MutatingUseContext::Call
                | MutatingUseContext::Yield
                | MutatingUseContext::AsmOutput,
            ) => place.is_indirect().then_some(DefUse::Use),

            // All other contexts are uses...
            PlaceContext::MutatingUse(
                MutatingUseContext::AddressOf
                | MutatingUseContext::Borrow
                | MutatingUseContext::Drop
                | MutatingUseContext::Retag,
            )
            | PlaceContext::NonMutatingUse(
                NonMutatingUseContext::AddressOf
                | NonMutatingUseContext::Copy
                | NonMutatingUseContext::Inspect
                | NonMutatingUseContext::Move
                | NonMutatingUseContext::ShallowBorrow
                | NonMutatingUseContext::SharedBorrow
                | NonMutatingUseContext::UniqueBorrow,
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
pub struct MaybeTransitiveLiveLocals<'a, 'tcx> {
    always_live: &'a BitSet<Local>,
    local_decls: &'a LocalDecls<'tcx>,
    tcx: TyCtxt<'tcx>,
}

impl<'a, 'tcx> MaybeTransitiveLiveLocals<'a, 'tcx> {
    /// The `always_alive` set is the set of locals to which all stores should unconditionally be
    /// considered live.
    ///
    /// This should include at least all locals that are ever borrowed.
    pub fn new(
        always_live: &'a BitSet<Local>,
        local_decls: &'a LocalDecls<'tcx>,
        tcx: TyCtxt<'tcx>,
    ) -> Self {
        MaybeTransitiveLiveLocals { always_live, local_decls, tcx }
    }
}

impl<'a, 'tcx> AnalysisDomain<'tcx> for MaybeTransitiveLiveLocals<'a, 'tcx> {
    type Domain = ChunkedBitSet<Local>;
    type Direction = Backward;

    const NAME: &'static str = "transitive liveness";

    fn bottom_value(&self, body: &mir::Body<'tcx>) -> Self::Domain {
        // bottom = not live
        ChunkedBitSet::new_empty(body.local_decls.len())
    }

    fn initialize_start_block(&self, _: &mir::Body<'tcx>, _: &mut Self::Domain) {
        // No variables are live until we observe a use
    }
}

struct TransferWrapper<'a>(&'a mut ChunkedBitSet<Local>);

impl<'a> GenKill<Local> for TransferWrapper<'a> {
    fn gen(&mut self, l: Local) {
        self.0.insert(l);
    }

    fn kill(&mut self, l: Local) {
        self.0.remove(l);
    }
}

impl<'a, 'tcx> Analysis<'tcx> for MaybeTransitiveLiveLocals<'a, 'tcx> {
    fn apply_statement_effect(
        &self,
        trans: &mut Self::Domain,
        statement: &mir::Statement<'tcx>,
        location: Location,
    ) {
        // Compute the place that we are storing to, if any
        let destination = match &statement.kind {
            StatementKind::Assign(assign) => {
                if assign.1.is_pointer_int_cast(self.local_decls, self.tcx) {
                    // Pointer to int casts may be side-effects due to exposing the provenance.
                    // While the model is undecided, we should be conservative. See
                    // <https://www.ralfj.de/blog/2022/04/11/provenance-exposed.html>
                    None
                } else {
                    Some(assign.0)
                }
            }
            StatementKind::SetDiscriminant { place, .. } | StatementKind::Deinit(place) => {
                Some(**place)
            }
            StatementKind::FakeRead(_)
            | StatementKind::StorageLive(_)
            | StatementKind::StorageDead(_)
            | StatementKind::Retag(..)
            | StatementKind::AscribeUserType(..)
            | StatementKind::Coverage(..)
            | StatementKind::CopyNonOverlapping(..)
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
        TransferFunction(&mut TransferWrapper(trans)).visit_statement(statement, location);
    }

    fn apply_terminator_effect(
        &self,
        trans: &mut Self::Domain,
        terminator: &mir::Terminator<'tcx>,
        location: Location,
    ) {
        TransferFunction(&mut TransferWrapper(trans)).visit_terminator(terminator, location);
    }

    fn apply_call_return_effect(
        &self,
        trans: &mut Self::Domain,
        _block: mir::BasicBlock,
        return_places: CallReturnPlaces<'_, 'tcx>,
    ) {
        return_places.for_each(|place| {
            if let Some(local) = place.as_local() {
                trans.remove(local);
            }
        });
    }

    fn apply_yield_resume_effect(
        &self,
        trans: &mut Self::Domain,
        _resume_block: mir::BasicBlock,
        resume_place: mir::Place<'tcx>,
    ) {
        if let Some(local) = resume_place.as_local() {
            trans.remove(local);
        }
    }
}
