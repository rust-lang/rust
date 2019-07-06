pub use super::*;

use rustc::mir::*;
use rustc::mir::visit::{
    PlaceContext, Visitor, NonMutatingUseContext,
};
use std::cell::RefCell;
use crate::dataflow::BitDenotation;
use crate::dataflow::HaveBeenBorrowedLocals;
use crate::dataflow::{DataflowResults, DataflowResultsCursor, DataflowResultsRefCursor};

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
        // Nothing is live on function entry (generators only have a self
        // argument, and we don't care about that)
        assert_eq!(1, self.body.arg_count);
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

/// Dataflow analysis that determines whether each local requires storage at a
/// given location; i.e. whether its storage can go away without being observed.
pub struct RequiresStorage<'mir, 'tcx> {
    body: &'mir Body<'tcx>,
    borrowed_locals:
        RefCell<DataflowResultsRefCursor<'mir, 'tcx, HaveBeenBorrowedLocals<'mir, 'tcx>>>,
}

impl<'mir, 'tcx: 'mir> RequiresStorage<'mir, 'tcx> {
    pub fn new(
        body: &'mir Body<'tcx>,
        borrowed_locals: &'mir DataflowResults<'tcx, HaveBeenBorrowedLocals<'mir, 'tcx>>,
    ) -> Self {
        RequiresStorage {
            body,
            borrowed_locals: RefCell::new(DataflowResultsCursor::new(borrowed_locals, body)),
        }
    }

    pub fn body(&self) -> &Body<'tcx> {
        self.body
    }
}

impl<'mir, 'tcx> BitDenotation<'tcx> for RequiresStorage<'mir, 'tcx> {
    type Idx = Local;
    fn name() -> &'static str { "requires_storage" }
    fn bits_per_block(&self) -> usize {
        self.body.local_decls.len()
    }

    fn start_block_effect(&self, _sets: &mut BitSet<Local>) {
        // Nothing is live on function entry (generators only have a self
        // argument, and we don't care about that)
        assert_eq!(1, self.body.arg_count);
    }

    fn statement_effect(&self,
                        sets: &mut GenKillSet<Local>,
                        loc: Location) {
        self.check_for_move(sets, loc);
        self.check_for_borrow(sets, loc);

        let stmt = &self.body[loc.block].statements[loc.statement_index];
        match stmt.kind {
            StatementKind::StorageLive(l) => sets.gen(l),
            StatementKind::StorageDead(l) => sets.kill(l),
            StatementKind::Assign(ref place, _)
            | StatementKind::SetDiscriminant { ref place, .. } => {
                place.base_local().map(|l| sets.gen(l));
            }
            StatementKind::InlineAsm(box InlineAsm { ref outputs, .. }) => {
                for p in &**outputs {
                    p.base_local().map(|l| sets.gen(l));
                }
            }
            _ => (),
        }
    }

    fn terminator_effect(&self,
                         sets: &mut GenKillSet<Local>,
                         loc: Location) {
        self.check_for_move(sets, loc);
        self.check_for_borrow(sets, loc);
    }

    fn propagate_call_return(
        &self,
        in_out: &mut BitSet<Local>,
        _call_bb: mir::BasicBlock,
        _dest_bb: mir::BasicBlock,
        dest_place: &mir::Place<'tcx>,
    ) {
        dest_place.base_local().map(|l| in_out.insert(l));
    }
}

impl<'mir, 'tcx> RequiresStorage<'mir, 'tcx> {
    /// Kill locals that are fully moved and have not been borrowed.
    fn check_for_move(&self, sets: &mut GenKillSet<Local>, loc: Location) {
        let mut visitor = MoveVisitor {
            sets,
            borrowed_locals: &self.borrowed_locals,
        };
        visitor.visit_location(self.body, loc);
    }

    /// Gen locals that are newly borrowed. This includes borrowing any part of
    /// a local (we rely on this behavior of `HaveBeenBorrowedLocals`).
    fn check_for_borrow(&self, sets: &mut GenKillSet<Local>, loc: Location) {
        let mut borrowed_locals = self.borrowed_locals.borrow_mut();
        borrowed_locals.seek(loc);
        borrowed_locals.each_gen_bit(|l| sets.gen(l));
    }
}

impl<'mir, 'tcx> BottomValue for RequiresStorage<'mir, 'tcx> {
    /// bottom = dead
    const BOTTOM_VALUE: bool = false;
}

struct MoveVisitor<'a, 'mir, 'tcx> {
    borrowed_locals:
        &'a RefCell<DataflowResultsRefCursor<'mir, 'tcx, HaveBeenBorrowedLocals<'mir, 'tcx>>>,
    sets: &'a mut GenKillSet<Local>,
}

impl<'a, 'mir: 'a, 'tcx> Visitor<'tcx> for MoveVisitor<'a, 'mir, 'tcx> {
    fn visit_local(&mut self, local: &Local, context: PlaceContext, loc: Location) {
        if PlaceContext::NonMutatingUse(NonMutatingUseContext::Move) == context {
            let mut borrowed_locals = self.borrowed_locals.borrow_mut();
            borrowed_locals.seek(loc);
            if !borrowed_locals.contains(*local) {
                self.sets.kill(*local);
            }
        }
    }
}
