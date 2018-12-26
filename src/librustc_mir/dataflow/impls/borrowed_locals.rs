pub use super::*;

use rustc::mir::*;
use rustc::mir::visit::Visitor;
use dataflow::BitDenotation;

/// This calculates if any part of a MIR local could have previously been borrowed.
/// This means that once a local has been borrowed, its bit will be set
/// from that point and onwards, until we see a StorageDead statement for the local,
/// at which points there is no memory associated with the local, so it cannot be borrowed.
/// This is used to compute which locals are live during a yield expression for
/// immovable generators.
#[derive(Copy, Clone)]
pub struct HaveBeenBorrowedLocals<'a, 'tcx: 'a> {
    mir: &'a Mir<'tcx>,
}

impl<'a, 'tcx: 'a> HaveBeenBorrowedLocals<'a, 'tcx> {
    pub fn new(mir: &'a Mir<'tcx>)
               -> Self {
        HaveBeenBorrowedLocals { mir }
    }

    pub fn mir(&self) -> &Mir<'tcx> {
        self.mir
    }
}

impl<'a, 'tcx> BitDenotation<'tcx> for HaveBeenBorrowedLocals<'a, 'tcx> {
    type Idx = Local;
    fn name() -> &'static str { "has_been_borrowed_locals" }
    fn bits_per_block(&self) -> usize {
        self.mir.local_decls.len()
    }

    fn start_block_effect(&self, _sets: &mut BitSet<Local>) {
        // Nothing is borrowed on function entry
    }

    fn statement_effect(&self,
                        sets: &mut BlockSets<Local>,
                        loc: Location) {
        let stmt = &self.mir[loc.block].statements[loc.statement_index];

        BorrowedLocalsVisitor {
            sets,
        }.visit_statement(loc.block, stmt, loc);

        // StorageDead invalidates all borrows and raw pointers to a local
        match stmt.kind {
            StatementKind::StorageDead(l) => sets.kill(l),
            _ => (),
        }
    }

    fn terminator_effect(&self,
                         sets: &mut BlockSets<Local>,
                         loc: Location) {
        BorrowedLocalsVisitor {
            sets,
        }.visit_terminator(loc.block, self.mir[loc.block].terminator(), loc);
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

impl<'a, 'tcx> BitSetOperator for HaveBeenBorrowedLocals<'a, 'tcx> {
    #[inline]
    fn join<T: Idx>(&self, inout_set: &mut BitSet<T>, in_set: &BitSet<T>) -> bool {
        inout_set.union(in_set) // "maybe" means we union effects of both preds
    }
}

impl<'a, 'tcx> InitialFlow for HaveBeenBorrowedLocals<'a, 'tcx> {
    #[inline]
    fn bottom_value() -> bool {
        false // bottom = unborrowed
    }
}

struct BorrowedLocalsVisitor<'b, 'c: 'b> {
    sets: &'b mut BlockSets<'c, Local>,
}

fn find_local<'tcx>(place: &Place<'tcx>) -> Option<Local> {
    match *place {
        Place::Local(l) => Some(l),
        Place::Promoted(_) |
        Place::Static(..) => None,
        Place::Projection(ref proj) => {
            match proj.elem {
                ProjectionElem::Deref => None,
                _ => find_local(&proj.base)
            }
        }
    }
}

impl<'tcx, 'b, 'c> Visitor<'tcx> for BorrowedLocalsVisitor<'b, 'c> {
    fn visit_rvalue(&mut self,
                    rvalue: &Rvalue<'tcx>,
                    location: Location) {
        if let Rvalue::Ref(_, _, ref place) = *rvalue {
            if let Some(local) = find_local(place) {
                self.sets.gen(local);
            }
        }

        self.super_rvalue(rvalue, location)
    }
}
