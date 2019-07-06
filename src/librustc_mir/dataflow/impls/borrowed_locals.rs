pub use super::*;

use rustc::mir::*;
use rustc::mir::visit::Visitor;
use crate::dataflow::{BitDenotation, GenKillSet};

/// This calculates if any part of a MIR local could have previously been borrowed.
/// This means that once a local has been borrowed, its bit will be set
/// from that point and onwards, until we see a StorageDead statement for the local,
/// at which points there is no memory associated with the local, so it cannot be borrowed.
/// This is used to compute which locals are live during a yield expression for
/// immovable generators.
#[derive(Copy, Clone)]
pub struct HaveBeenBorrowedLocals<'a, 'tcx> {
    body: &'a Body<'tcx>,
}

impl<'a, 'tcx> HaveBeenBorrowedLocals<'a, 'tcx> {
    pub fn new(body: &'a Body<'tcx>)
               -> Self {
        HaveBeenBorrowedLocals { body }
    }

    pub fn body(&self) -> &Body<'tcx> {
        self.body
    }
}

impl<'a, 'tcx> BitDenotation<'tcx> for HaveBeenBorrowedLocals<'a, 'tcx> {
    type Idx = Local;
    fn name() -> &'static str { "has_been_borrowed_locals" }
    fn bits_per_block(&self) -> usize {
        self.body.local_decls.len()
    }

    fn start_block_effect(&self, _on_entry: &mut BitSet<Local>) {
        // Nothing is borrowed on function entry
    }

    fn statement_effect(&self,
                        trans: &mut GenKillSet<Local>,
                        loc: Location) {
        let stmt = &self.body[loc.block].statements[loc.statement_index];

        BorrowedLocalsVisitor {
            trans,
        }.visit_statement(stmt, loc);

        // StorageDead invalidates all borrows and raw pointers to a local
        match stmt.kind {
            StatementKind::StorageDead(l) => trans.kill(l),
            _ => (),
        }
    }

    fn terminator_effect(&self,
                         trans: &mut GenKillSet<Local>,
                         loc: Location) {
        let terminator = self.body[loc.block].terminator();
        BorrowedLocalsVisitor {
            trans,
        }.visit_terminator(terminator, loc);
        match &terminator.kind {
            // Drop terminators borrows the location
            TerminatorKind::Drop { location, .. } |
            TerminatorKind::DropAndReplace { location, .. } => {
                if let Some(local) = find_local(location) {
                    trans.gen(local);
                }
            }
            _ => (),
        }
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

impl<'a, 'tcx> BottomValue for HaveBeenBorrowedLocals<'a, 'tcx> {
    // bottom = unborrowed
    const BOTTOM_VALUE: bool = false;
}

struct BorrowedLocalsVisitor<'gk> {
    trans: &'gk mut GenKillSet<Local>,
}

fn find_local(place: &Place<'_>) -> Option<Local> {
    place.iterate(|place_base, place_projection| {
        for proj in place_projection {
            if proj.elem == ProjectionElem::Deref {
                return None;
            }
        }

        if let PlaceBase::Local(local) = place_base {
            Some(*local)
        } else {
            None
        }
    })
}

impl<'tcx> Visitor<'tcx> for BorrowedLocalsVisitor<'_> {
    fn visit_rvalue(&mut self,
                    rvalue: &Rvalue<'tcx>,
                    location: Location) {
        if let Rvalue::Ref(_, _, ref place) = *rvalue {
            if let Some(local) = find_local(place) {
                self.trans.gen(local);
            }
        }

        self.super_rvalue(rvalue, location)
    }
}
