pub use super::*;

use rustc::mir::*;
use rustc::mir::visit::{
    PlaceContext, Visitor, NonMutatingUseContext,
};
use crate::dataflow::BitDenotation;
use crate::dataflow::HaveBeenBorrowedLocals;
use crate::dataflow::{DataflowResults, state_for_location};

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
#[derive(Copy, Clone)]
pub struct RequiresStorage<'a: 'b, 'b, 'tcx: 'a> {
    body: &'a Body<'tcx>,
    borrowed_locals: Option<&'b (HaveBeenBorrowedLocals<'a, 'tcx>,
                                 DataflowResults<'tcx, HaveBeenBorrowedLocals<'a, 'tcx>>)>,
}

impl<'a, 'b, 'tcx: 'a> RequiresStorage<'a, 'b, 'tcx> {
    pub fn new(
        body: &'a Body<'tcx>,
        borrowed_locals: Option<&'b (HaveBeenBorrowedLocals<'a, 'tcx>,
                                     DataflowResults<'tcx, HaveBeenBorrowedLocals<'a, 'tcx>>)>
    ) -> Self {
        RequiresStorage { body, borrowed_locals }
    }

    pub fn body(&self) -> &Body<'tcx> {
        self.body
    }
}

impl<'a, 'b, 'tcx> BitDenotation<'tcx> for RequiresStorage<'a, 'b, 'tcx> {
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

impl<'a, 'b, 'tcx> RequiresStorage<'a, 'b, 'tcx> {
    /// Kill locals that are fully moved and have not been borrowed.
    fn check_for_move(&self, sets: &mut BlockSets<'_, Local>, body: &Body<'tcx>, loc: Location) {
        // TODO avoid recomputing this every time.
        let borrowed_locals = match self.borrowed_locals {
            Some((ref analysis, ref results)) =>
                Some(state_for_location(loc, analysis, results, self.body)),
            None => None,
        };
        let mut visitor = MoveVisitor { sets, borrowed_locals };
        visitor.visit_location(body, loc);
    }
}

impl<'a, 'b, 'tcx> BitSetOperator for RequiresStorage<'a, 'b, 'tcx> {
    #[inline]
    fn join<T: Idx>(&self, inout_set: &mut BitSet<T>, in_set: &BitSet<T>) -> bool {
        inout_set.union(in_set) // "maybe" means we union effects of both preds
    }
}

impl<'a, 'b, 'tcx> InitialFlow for RequiresStorage<'a, 'b, 'tcx> {
    #[inline]
    fn bottom_value() -> bool {
        false // bottom = dead
    }
}

struct MoveVisitor<'a, 'b> {
    sets: &'a mut BlockSets<'b, Local>,
    borrowed_locals: Option<BitSet<Local>>,
}

impl<'a, 'b, 'tcx> Visitor<'tcx> for MoveVisitor<'a, 'b> {
    fn visit_local(&mut self, local: &Local, context: PlaceContext, loc: Location) {
        if let Some(ref borrowed_locals) = self.borrowed_locals {
            if borrowed_locals.contains(*local) {
                return;
            }
        }
        if PlaceContext::NonMutatingUse(NonMutatingUseContext::Move) == context {
            self.sets.kill(*local);
        }
        self.super_local(local, context, loc);
    }
}
