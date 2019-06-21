pub use super::*;

use rustc::mir::*;
use crate::dataflow::BitDenotation;

#[derive(Copy, Clone)]
pub struct MaybeStorageLive<'a, 'tcx> {
    body: &'a Body<'tcx>,
}

impl<'a, 'tcx> MaybeStorageLive<'a, 'tcx> {
    pub fn new(body: &'a Body<'tcx>)
               -> Self {
        MaybeStorageLive { body }
    }

    pub fn body(&self) -> &Body<'tcx> {
        self.body
    }
}

impl<'a, 'tcx> BitDenotation<'tcx> for MaybeStorageLive<'a, 'tcx> {
    type Idx = Local;
    fn name() -> &'static str { "maybe_storage_live" }
    fn bits_per_block(&self) -> usize {
        self.body.local_decls.len()
    }

    fn start_block_effect(&self, _on_entry: &mut BitSet<Local>) {
        // Nothing is live on function entry
    }

    fn statement_effect(&self,
                        trans: &mut GenKillSet<Local>,
                        loc: Location) {
        let stmt = &self.body[loc.block].statements[loc.statement_index];

        match stmt.kind {
            StatementKind::StorageLive(l) => trans.gen(l),
            StatementKind::StorageDead(l) => trans.kill(l),
            _ => (),
        }
    }

    fn terminator_effect(&self,
                         _trans: &mut GenKillSet<Local>,
                         _loc: Location) {
        // Terminators have no effect
    }

    fn propagate_call_return(
        &self,
        _in_out: &mut BitSet<Local>,
        _call_bb: mir::BasicBlock,
        _dest_bb: mir::BasicBlock,
        _dest_place: &mir::Place<'tcx>,
    ) {
        // Nothing to do when a call returns successfully
    }
}

impl<'a, 'tcx> BottomValue for MaybeStorageLive<'a, 'tcx> {
    /// bottom = dead
    const BOTTOM_VALUE: bool = false;
}
