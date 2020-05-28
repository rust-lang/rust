pub use super::*;

use crate::dataflow::BottomValue;
use crate::dataflow::{self, GenKill};
use crate::util::storage::AlwaysLiveLocals;
use rustc_middle::mir::*;

#[derive(Clone)]
pub struct MaybeStorageLive {
    always_live_locals: AlwaysLiveLocals,
}

impl MaybeStorageLive {
    pub fn new(always_live_locals: AlwaysLiveLocals) -> Self {
        MaybeStorageLive { always_live_locals }
    }
}

impl dataflow::AnalysisDomain<'tcx> for MaybeStorageLive {
    type Idx = Local;

    const NAME: &'static str = "maybe_storage_live";

    fn bits_per_block(&self, body: &mir::Body<'tcx>) -> usize {
        body.local_decls.len()
    }

    fn initialize_start_block(&self, body: &mir::Body<'tcx>, on_entry: &mut BitSet<Self::Idx>) {
        assert_eq!(body.local_decls.len(), self.always_live_locals.domain_size());
        for local in self.always_live_locals.iter() {
            on_entry.insert(local);
        }

        for arg in body.args_iter() {
            on_entry.insert(arg);
        }
    }
}

impl dataflow::GenKillAnalysis<'tcx> for MaybeStorageLive {
    fn statement_effect(
        &self,
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
        &self,
        _trans: &mut impl GenKill<Self::Idx>,
        _: &mir::Terminator<'tcx>,
        _: Location,
    ) {
        // Terminators have no effect
    }

    fn call_return_effect(
        &self,
        _trans: &mut impl GenKill<Self::Idx>,
        _block: BasicBlock,
        _func: &mir::Operand<'tcx>,
        _args: &[mir::Operand<'tcx>],
        _return_place: mir::Place<'tcx>,
    ) {
        // Nothing to do when a call returns successfully
    }
}

impl BottomValue for MaybeStorageLive {
    /// bottom = dead
    const BOTTOM_VALUE: bool = false;
}
