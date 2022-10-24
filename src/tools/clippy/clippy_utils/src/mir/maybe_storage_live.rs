use rustc_index::bit_set::BitSet;
use rustc_middle::mir;
use rustc_mir_dataflow::{AnalysisDomain, CallReturnPlaces, GenKill, GenKillAnalysis};

/// Determines liveness of each local purely based on `StorageLive`/`Dead`.
#[derive(Copy, Clone)]
pub(super) struct MaybeStorageLive;

impl<'tcx> AnalysisDomain<'tcx> for MaybeStorageLive {
    type Domain = BitSet<mir::Local>;
    const NAME: &'static str = "maybe_storage_live";

    fn bottom_value(&self, body: &mir::Body<'tcx>) -> Self::Domain {
        // bottom = dead
        BitSet::new_empty(body.local_decls.len())
    }

    fn initialize_start_block(&self, body: &mir::Body<'tcx>, state: &mut Self::Domain) {
        for arg in body.args_iter() {
            state.insert(arg);
        }
    }
}

impl<'tcx> GenKillAnalysis<'tcx> for MaybeStorageLive {
    type Idx = mir::Local;

    fn statement_effect(&self, trans: &mut impl GenKill<Self::Idx>, stmt: &mir::Statement<'tcx>, _: mir::Location) {
        match stmt.kind {
            mir::StatementKind::StorageLive(l) => trans.gen(l),
            mir::StatementKind::StorageDead(l) => trans.kill(l),
            _ => (),
        }
    }

    fn terminator_effect(
        &self,
        _trans: &mut impl GenKill<Self::Idx>,
        _terminator: &mir::Terminator<'tcx>,
        _loc: mir::Location,
    ) {
    }

    fn call_return_effect(
        &self,
        _trans: &mut impl GenKill<Self::Idx>,
        _block: mir::BasicBlock,
        _return_places: CallReturnPlaces<'_, 'tcx>,
    ) {
        // Nothing to do when a call returns successfully
    }
}
