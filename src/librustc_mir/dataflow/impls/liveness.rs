use rustc_index::bit_set::BitSet;
use rustc_middle::mir::visit::{MutatingUseContext, NonMutatingUseContext, PlaceContext, Visitor};
use rustc_middle::mir::{self, Local, Location};

use crate::dataflow::{AnalysisDomain, Backward, BottomValue, GenKill, GenKillAnalysis};

/// A [live-variable dataflow analysis][liveness].
///
/// [liveness]: https://en.wikipedia.org/wiki/Live_variable_analysis
pub struct MaybeLiveLocals;

impl MaybeLiveLocals {
    fn transfer_function<T>(&self, trans: &'a mut T) -> TransferFunction<'a, T> {
        TransferFunction(trans)
    }
}

impl BottomValue for MaybeLiveLocals {
    // bottom = not live
    const BOTTOM_VALUE: bool = false;
}

impl AnalysisDomain<'tcx> for MaybeLiveLocals {
    type Idx = Local;
    type Direction = Backward;

    const NAME: &'static str = "liveness";

    fn bits_per_block(&self, body: &mir::Body<'tcx>) -> usize {
        body.local_decls.len()
    }

    fn initialize_start_block(&self, _: &mir::Body<'tcx>, _: &mut BitSet<Self::Idx>) {
        // No variables are live until we observe a use
    }
}

impl GenKillAnalysis<'tcx> for MaybeLiveLocals {
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
    fn visit_local(&mut self, &local: &Local, context: PlaceContext, _: Location) {
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
            // only a `Def` when the function returns succesfully, we handle this case separately
            // in `call_return_effect` above.
            PlaceContext::MutatingUse(MutatingUseContext::Call | MutatingUseContext::Yield) => None,

            // All other contexts are uses...
            PlaceContext::MutatingUse(
                MutatingUseContext::AddressOf
                | MutatingUseContext::AsmOutput
                | MutatingUseContext::Borrow
                | MutatingUseContext::Drop
                | MutatingUseContext::Projection
                | MutatingUseContext::Retag,
            )
            | PlaceContext::NonMutatingUse(
                NonMutatingUseContext::AddressOf
                | NonMutatingUseContext::Copy
                | NonMutatingUseContext::Inspect
                | NonMutatingUseContext::Move
                | NonMutatingUseContext::Projection
                | NonMutatingUseContext::ShallowBorrow
                | NonMutatingUseContext::SharedBorrow
                | NonMutatingUseContext::UniqueBorrow,
            ) => Some(DefUse::Use),
        }
    }
}
