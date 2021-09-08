use rustc_index::bit_set::BitSet;
use rustc_middle::mir::visit::{MutatingUseContext, NonMutatingUseContext, PlaceContext, Visitor};
use rustc_middle::mir::{self, Local, Location};

use crate::{AnalysisDomain, Backward, GenKill, GenKillAnalysis};

/// A [live-variable dataflow analysis][liveness].
///
/// This analysis considers references as being used only at the point of the
/// borrow. In other words, this analysis does not track uses because of references that already
/// exist. See [this `mir-dataflow` test][flow-test] for an example. You almost never want to use
/// this analysis without also looking at the results of [`MaybeBorrowedLocals`].
///
/// [`MaybeBorrowedLocals`]: super::MaybeBorrowedLocals
/// [flow-test]: https://github.com/rust-lang/rust/blob/a08c47310c7d49cbdc5d7afb38408ba519967ecd/src/test/ui/mir-dataflow/liveness-ptr.rs
/// [liveness]: https://en.wikipedia.org/wiki/Live_variable_analysis
pub struct MaybeLiveLocals;

impl MaybeLiveLocals {
    fn transfer_function<T>(&self, trans: &'a mut T) -> TransferFunction<'a, T> {
        TransferFunction(trans)
    }
}

impl AnalysisDomain<'tcx> for MaybeLiveLocals {
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

impl GenKillAnalysis<'tcx> for MaybeLiveLocals {
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
        _func: &mir::Operand<'tcx>,
        _args: &[mir::Operand<'tcx>],
        dest_place: mir::Place<'tcx>,
    ) {
        if let Some(local) = dest_place.as_local() {
            trans.kill(local);
        }
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
        let mir::Place { projection, local } = *place;

        // We purposefully do not call `super_place` here to avoid calling `visit_local` for this
        // place with one of the `Projection` variants of `PlaceContext`.
        self.visit_projection(place.as_ref(), context, location);

        match DefUse::for_place(context) {
            // Treat derefs as a use of the base local. `*p = 4` is not a def of `p` but a use.
            Some(_) if place.is_indirect() => self.0.gen(local),

            Some(DefUse::Def) if projection.is_empty() => self.0.kill(local),
            Some(DefUse::Use) => self.0.gen(local),
            _ => {}
        }
    }

    fn visit_local(&mut self, &local: &Local, context: PlaceContext, _: Location) {
        // Because we do not call `super_place` above, `visit_local` is only called for locals that
        // do not appear as part of  a `Place` in the MIR. This handles cases like the implicit use
        // of the return place in a `Return` terminator or the index in an `Index` projection.
        match DefUse::for_place(context) {
            Some(DefUse::Def) => self.0.kill(local),
            Some(DefUse::Use) => self.0.gen(local),
            _ => {}
        }
    }
}

#[derive(Eq, PartialEq, Clone)]
enum DefUse {
    Def,
    Use,
}

impl DefUse {
    fn for_place(context: PlaceContext) -> Option<DefUse> {
        match context {
            PlaceContext::NonUse(_) => None,

            PlaceContext::MutatingUse(MutatingUseContext::Store) => Some(DefUse::Def),

            // `MutatingUseContext::Call` and `MutatingUseContext::Yield` indicate that this is the
            // destination place for a `Call` return or `Yield` resume respectively. Since this is
            // only a `Def` when the function returns successfully, we handle this case separately
            // in `call_return_effect` above.
            PlaceContext::MutatingUse(MutatingUseContext::Call | MutatingUseContext::Yield) => None,

            // All other contexts are uses...
            PlaceContext::MutatingUse(
                MutatingUseContext::AddressOf
                | MutatingUseContext::AsmOutput
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
