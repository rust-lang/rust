//! A less precise version of `MaybeInitializedPlaces` whose domain is entire locals.
//!
//! A local will be maybe initialized if *any* projections of that local might be initialized.

use crate::{CallReturnPlaces, GenKill};

use rustc_index::bit_set::BitSet;
use rustc_middle::mir::visit::{PlaceContext, Visitor};
use rustc_middle::mir::{self, BasicBlock, Local, Location};

pub struct MaybeInitializedLocals;

impl<'tcx> crate::AnalysisDomain<'tcx> for MaybeInitializedLocals {
    type Domain = BitSet<Local>;

    const NAME: &'static str = "maybe_init_locals";

    fn bottom_value(&self, body: &mir::Body<'tcx>) -> Self::Domain {
        // bottom = uninit
        BitSet::new_empty(body.local_decls.len())
    }

    fn initialize_start_block(&self, body: &mir::Body<'tcx>, entry_set: &mut Self::Domain) {
        // Function arguments are initialized to begin with.
        for arg in body.args_iter() {
            entry_set.insert(arg);
        }
    }
}

impl<'tcx> crate::GenKillAnalysis<'tcx> for MaybeInitializedLocals {
    type Idx = Local;

    fn statement_effect(
        &self,
        trans: &mut impl GenKill<Self::Idx>,
        statement: &mir::Statement<'tcx>,
        loc: Location,
    ) {
        TransferFunction { trans }.visit_statement(statement, loc)
    }

    fn terminator_effect(
        &self,
        trans: &mut impl GenKill<Self::Idx>,
        terminator: &mir::Terminator<'tcx>,
        loc: Location,
    ) {
        TransferFunction { trans }.visit_terminator(terminator, loc)
    }

    fn call_return_effect(
        &self,
        trans: &mut impl GenKill<Self::Idx>,
        _block: BasicBlock,
        return_places: CallReturnPlaces<'_, 'tcx>,
    ) {
        return_places.for_each(|place| trans.gen(place.local));
    }

    /// See `Analysis::apply_yield_resume_effect`.
    fn yield_resume_effect(
        &self,
        trans: &mut impl GenKill<Self::Idx>,
        _resume_block: BasicBlock,
        resume_place: mir::Place<'tcx>,
    ) {
        trans.gen(resume_place.local)
    }
}

struct TransferFunction<'a, T> {
    trans: &'a mut T,
}

impl<T> Visitor<'_> for TransferFunction<'_, T>
where
    T: GenKill<Local>,
{
    fn visit_local(&mut self, &local: &Local, context: PlaceContext, _: Location) {
        use rustc_middle::mir::visit::{MutatingUseContext, NonMutatingUseContext, NonUseContext};
        match context {
            // These are handled specially in `call_return_effect` and `yield_resume_effect`.
            PlaceContext::MutatingUse(
                MutatingUseContext::Call
                | MutatingUseContext::AsmOutput
                | MutatingUseContext::Yield,
            ) => {}

            // Otherwise, when a place is mutated, we must consider it possibly initialized.
            PlaceContext::MutatingUse(_) => self.trans.gen(local),

            // If the local is moved out of, or if it gets marked `StorageDead`, consider it no
            // longer initialized.
            PlaceContext::NonUse(NonUseContext::StorageDead)
            | PlaceContext::NonMutatingUse(NonMutatingUseContext::Move) => self.trans.kill(local),

            // All other uses do not affect this analysis.
            PlaceContext::NonUse(
                NonUseContext::StorageLive
                | NonUseContext::AscribeUserTy
                | NonUseContext::VarDebugInfo,
            )
            | PlaceContext::NonMutatingUse(
                NonMutatingUseContext::Inspect
                | NonMutatingUseContext::Copy
                | NonMutatingUseContext::SharedBorrow
                | NonMutatingUseContext::ShallowBorrow
                | NonMutatingUseContext::UniqueBorrow
                | NonMutatingUseContext::AddressOf
                | NonMutatingUseContext::Projection,
            ) => {}
        }
    }
}
