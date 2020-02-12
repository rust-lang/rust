pub use super::*;

use crate::dataflow::generic::{Results, ResultsRefCursor};
use crate::dataflow::BitDenotation;
use crate::dataflow::MaybeBorrowedLocals;
use rustc::mir::visit::{NonMutatingUseContext, PlaceContext, Visitor};
use rustc::mir::*;
use std::cell::RefCell;

#[derive(Copy, Clone)]
pub struct MaybeStorageLive<'a, 'tcx> {
    body: &'a Body<'tcx>,
}

impl<'a, 'tcx> MaybeStorageLive<'a, 'tcx> {
    pub fn new(body: &'a Body<'tcx>) -> Self {
        MaybeStorageLive { body }
    }

    pub fn body(&self) -> &Body<'tcx> {
        self.body
    }
}

impl<'a, 'tcx> BitDenotation<'tcx> for MaybeStorageLive<'a, 'tcx> {
    type Idx = Local;
    fn name() -> &'static str {
        "maybe_storage_live"
    }
    fn bits_per_block(&self) -> usize {
        self.body.local_decls.len()
    }

    fn start_block_effect(&self, on_entry: &mut BitSet<Local>) {
        // The resume argument is live on function entry (we don't care about
        // the `self` argument)
        for arg in self.body.args_iter().skip(1) {
            on_entry.insert(arg);
        }
    }

    fn statement_effect(&self, trans: &mut GenKillSet<Local>, loc: Location) {
        let stmt = &self.body[loc.block].statements[loc.statement_index];

        match stmt.kind {
            StatementKind::StorageLive(l) => trans.gen(l),
            StatementKind::StorageDead(l) => trans.kill(l),
            _ => (),
        }
    }

    fn terminator_effect(&self, _trans: &mut GenKillSet<Local>, _loc: Location) {
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

type BorrowedLocalsResults<'a, 'tcx> = ResultsRefCursor<'a, 'a, 'tcx, MaybeBorrowedLocals>;

/// Dataflow analysis that determines whether each local requires storage at a
/// given location; i.e. whether its storage can go away without being observed.
pub struct RequiresStorage<'mir, 'tcx> {
    body: ReadOnlyBodyAndCache<'mir, 'tcx>,
    borrowed_locals: RefCell<BorrowedLocalsResults<'mir, 'tcx>>,
}

impl<'mir, 'tcx: 'mir> RequiresStorage<'mir, 'tcx> {
    pub fn new(
        body: ReadOnlyBodyAndCache<'mir, 'tcx>,
        borrowed_locals: &'mir Results<'tcx, MaybeBorrowedLocals>,
    ) -> Self {
        RequiresStorage {
            body,
            borrowed_locals: RefCell::new(ResultsRefCursor::new(*body, borrowed_locals)),
        }
    }

    pub fn body(&self) -> &Body<'tcx> {
        &self.body
    }
}

impl<'mir, 'tcx> BitDenotation<'tcx> for RequiresStorage<'mir, 'tcx> {
    type Idx = Local;
    fn name() -> &'static str {
        "requires_storage"
    }
    fn bits_per_block(&self) -> usize {
        self.body.local_decls.len()
    }

    fn start_block_effect(&self, on_entry: &mut BitSet<Local>) {
        // The resume argument is live on function entry (we don't care about
        // the `self` argument)
        for arg in self.body.args_iter().skip(1) {
            on_entry.insert(arg);
        }
    }

    fn before_statement_effect(&self, sets: &mut GenKillSet<Self::Idx>, loc: Location) {
        let stmt = &self.body[loc.block].statements[loc.statement_index];

        // If a place is borrowed in a statement, it needs storage for that statement.
        self.borrowed_locals.borrow().analysis().statement_effect(sets, stmt, loc);

        // If a place is assigned to in a statement, it needs storage for that statement.
        match stmt.kind {
            StatementKind::StorageDead(l) => sets.kill(l),
            StatementKind::Assign(box (ref place, _))
            | StatementKind::SetDiscriminant { box ref place, .. } => {
                sets.gen(place.local);
            }
            StatementKind::InlineAsm(box InlineAsm { ref outputs, .. }) => {
                for place in &**outputs {
                    sets.gen(place.local);
                }
            }
            _ => (),
        }
    }

    fn statement_effect(&self, sets: &mut GenKillSet<Local>, loc: Location) {
        // If we move from a place then only stops needing storage *after*
        // that statement.
        self.check_for_move(sets, loc);
    }

    fn before_terminator_effect(&self, sets: &mut GenKillSet<Local>, loc: Location) {
        let terminator = self.body[loc.block].terminator();

        // If a place is borrowed in a terminator, it needs storage for that terminator.
        self.borrowed_locals.borrow().analysis().terminator_effect(sets, terminator, loc);

        if let TerminatorKind::Call { destination: Some((place, _)), .. } = terminator.kind {
            sets.gen(place.local);
        }
    }

    fn terminator_effect(&self, sets: &mut GenKillSet<Local>, loc: Location) {
        // For call terminators the destination requires storage for the call
        // and after the call returns successfully, but not after a panic.
        // Since `propagate_call_unwind` doesn't exist, we have to kill the
        // destination here, and then gen it again in `propagate_call_return`.
        if let TerminatorKind::Call { destination: Some((ref place, _)), .. } =
            self.body[loc.block].terminator().kind
        {
            if let Some(local) = place.as_local() {
                sets.kill(local);
            }
        }
        self.check_for_move(sets, loc);
    }

    fn propagate_call_return(
        &self,
        in_out: &mut BitSet<Local>,
        _call_bb: mir::BasicBlock,
        _dest_bb: mir::BasicBlock,
        dest_place: &mir::Place<'tcx>,
    ) {
        in_out.insert(dest_place.local);
    }
}

impl<'mir, 'tcx> RequiresStorage<'mir, 'tcx> {
    /// Kill locals that are fully moved and have not been borrowed.
    fn check_for_move(&self, sets: &mut GenKillSet<Local>, loc: Location) {
        let mut visitor = MoveVisitor { sets, borrowed_locals: &self.borrowed_locals };
        visitor.visit_location(self.body, loc);
    }
}

impl<'mir, 'tcx> BottomValue for RequiresStorage<'mir, 'tcx> {
    /// bottom = dead
    const BOTTOM_VALUE: bool = false;
}

struct MoveVisitor<'a, 'mir, 'tcx> {
    borrowed_locals: &'a RefCell<BorrowedLocalsResults<'mir, 'tcx>>,
    sets: &'a mut GenKillSet<Local>,
}

impl<'a, 'mir: 'a, 'tcx> Visitor<'tcx> for MoveVisitor<'a, 'mir, 'tcx> {
    fn visit_local(&mut self, local: &Local, context: PlaceContext, loc: Location) {
        if PlaceContext::NonMutatingUse(NonMutatingUseContext::Move) == context {
            let mut borrowed_locals = self.borrowed_locals.borrow_mut();
            borrowed_locals.seek_before(loc);
            if !borrowed_locals.contains(*local) {
                self.sets.kill(*local);
            }
        }
    }
}
