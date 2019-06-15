pub use super::*;

use rustc::mir::*;
use rustc::mir::visit::{
    PlaceContext, Visitor, NonMutatingUseContext,
};
use std::cell::RefCell;
use crate::dataflow::BitDenotation;
use crate::dataflow::HaveBeenBorrowedLocals;
use crate::dataflow::DataflowResultsCursor;

#[derive(Copy, Clone)]
pub struct MaybeStorageLive<'a, 'tcx: 'a> {
    body: &'a Body<'tcx>,
}

impl<'a, 'tcx: 'a> MaybeStorageLive<'a, 'tcx> {
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

    fn start_block_effect(&self, _sets: &mut BitSet<Local>) {
        // Nothing is live on function entry
    }

    fn statement_effect(&self,
                        sets: &mut BlockSets<'_, Local>,
                        loc: Location) {
        let stmt = &self.body[loc.block].statements[loc.statement_index];

        match stmt.kind {
            StatementKind::StorageLive(l) => sets.gen(l),
            StatementKind::StorageDead(l) => sets.kill(l),
            _ => (),
        }
    }

    fn terminator_effect(&self,
                         _sets: &mut BlockSets<'_, Local>,
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

impl<'a, 'tcx> BitSetOperator for MaybeStorageLive<'a, 'tcx> {
    #[inline]
    fn join<T: Idx>(&self, inout_set: &mut BitSet<T>, in_set: &BitSet<T>) -> bool {
        inout_set.union(in_set) // "maybe" means we union effects of both preds
    }
}

impl<'a, 'tcx> InitialFlow for MaybeStorageLive<'a, 'tcx> {
    #[inline]
    fn bottom_value() -> bool {
        false // bottom = dead
    }
}

/// Dataflow analysis that determines whether each local requires storage at a
/// given location; i.e. whether its storage can go away without being observed.
///
/// In the case of a movable generator, borrowed_locals can be `None` and we
/// will not consider borrows in this pass. This relies on the fact that we only
/// use this pass at yield points for these generators.
#[derive(Copy, Clone)]
pub struct RequiresStorage<'mir, 'tcx: 'mir, 'b> {
    body: &'mir Body<'tcx>,
    //borrowed_locals: Option<&'b (HaveBeenBorrowedLocals<'mir, 'tcx>,
    //                             DataflowResults<'tcx, HaveBeenBorrowedLocals<'mir, 'tcx>>)>,
    borrowed_locals:
        Option<&'b RefCell<DataflowResultsCursor<'mir, 'tcx, HaveBeenBorrowedLocals<'mir, 'tcx>>>>,
}

impl<'mir, 'tcx: 'mir, 'b> RequiresStorage<'mir, 'tcx, 'b> {
    pub fn new(
        body: &'mir Body<'tcx>,
        borrowed_locals: Option<&'b RefCell<DataflowResultsCursor<'mir, 'tcx, HaveBeenBorrowedLocals<'mir, 'tcx>>>>,
    ) -> Self {
        RequiresStorage { body, borrowed_locals }
    }

    pub fn body(&self) -> &Body<'tcx> {
        self.body
    }
}

impl<'mir, 'tcx, 'b> BitDenotation<'tcx> for RequiresStorage<'mir, 'tcx, 'b> {
    type Idx = Local;
    fn name() -> &'static str { "requires_storage" }
    fn bits_per_block(&self) -> usize {
        self.body.local_decls.len()
    }

    fn start_block_effect(&self, _sets: &mut BitSet<Local>) {
        // Nothing is live on function entry
    }

    fn statement_effect(&self,
                        sets: &mut BlockSets<'_, Local>,
                        loc: Location) {
        self.check_for_move(sets, self.body, loc);

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
                         sets: &mut BlockSets<'_, Local>,
                         loc: Location) {
        self.check_for_move(sets, self.body, loc);
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

impl<'mir, 'tcx, 'b> RequiresStorage<'mir, 'tcx, 'b> {
    /// Kill locals that are fully moved and have not been borrowed.
    fn check_for_move(&self, sets: &mut BlockSets<'_, Local>, body: &Body<'tcx>, loc: Location) {
        let mut visitor = MoveVisitor {
            sets,
            borrowed_locals: self.borrowed_locals,
        };
        visitor.visit_location(body, loc);
    }
}

impl<'mir, 'tcx, 'b> BitSetOperator for RequiresStorage<'mir, 'tcx, 'b> {
    #[inline]
    fn join<T: Idx>(&self, inout_set: &mut BitSet<T>, in_set: &BitSet<T>) -> bool {
        inout_set.union(in_set) // "maybe" means we union effects of both preds
    }
}

impl<'mir, 'tcx, 'b> InitialFlow for RequiresStorage<'mir, 'tcx, 'b> {
    #[inline]
    fn bottom_value() -> bool {
        false // bottom = dead
    }
}

struct MoveVisitor<'a, 'b, 'mir: 'a, 'tcx: 'mir> {
    sets: &'a mut BlockSets<'b, Local>,
    borrowed_locals: Option<&'a RefCell<DataflowResultsCursor<'mir, 'tcx, HaveBeenBorrowedLocals<'mir, 'tcx>>>>,
}

impl<'a, 'b, 'mir: 'a, 'tcx: 'mir> Visitor<'tcx> for MoveVisitor<'a, 'b, 'mir, 'tcx> {
    fn visit_local(&mut self, local: &Local, context: PlaceContext, loc: Location) {
        if let Some(ref borrowed_locals) = self.borrowed_locals {
            let mut borrowed_locals = borrowed_locals.borrow_mut();
            borrowed_locals.reconstruct_effect(loc);
            if borrowed_locals.contains(*local) {
                return;
            }
            // TODO check for newly borrowed locals and gen them
        }
        if PlaceContext::NonMutatingUse(NonMutatingUseContext::Move) == context {
            self.sets.kill(*local);
        }
    }
}
