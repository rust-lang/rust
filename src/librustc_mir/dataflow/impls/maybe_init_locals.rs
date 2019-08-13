pub use super::*;

use crate::dataflow::{BitDenotation, GenKillSet};
use rustc::mir::visit::{
    MutatingUseContext, NonMutatingUseContext, NonUseContext, PlaceContext, Visitor,
};
use rustc::mir::*;

/// This calculates if any part of a MIR local may still be initialized.
/// This means that once a local has been written to, its bit will be set
/// from that point and onwards, until we see an Operand::Move out of, or
/// StorageDead statement for the local.
/// This is an approximation of `MaybeInitializedPlaces` for whole locals.
#[derive(Copy, Clone)]
pub struct MaybeInitializedLocals<'a, 'tcx> {
    body: &'a Body<'tcx>,
}

impl<'a, 'tcx> MaybeInitializedLocals<'a, 'tcx> {
    pub fn new(body: &'a Body<'tcx>) -> Self {
        MaybeInitializedLocals { body }
    }

    pub fn body(&self) -> &Body<'tcx> {
        self.body
    }
}

impl<'a, 'tcx> BitDenotation<'tcx> for MaybeInitializedLocals<'a, 'tcx> {
    type Idx = Local;
    fn name() -> &'static str {
        "has_been_borrowed_locals"
    }
    fn bits_per_block(&self) -> usize {
        self.body.local_decls.len()
    }

    fn start_block_effect(&self, on_entry: &mut BitSet<Local>) {
        // Arguments are always initialized on entry.
        for arg in self.body.args_iter() {
            on_entry.insert(arg);
        }
    }

    fn statement_effect(&self, trans: &mut GenKillSet<Local>, loc: Location) {
        let stmt = &self.body[loc.block].statements[loc.statement_index];
        MaybeInitializedLocalsVisitor { trans }.visit_statement(stmt, loc);
    }

    fn terminator_effect(&self, trans: &mut GenKillSet<Local>, loc: Location) {
        let terminator = self.body[loc.block].terminator();
        MaybeInitializedLocalsVisitor { trans }.visit_terminator(terminator, loc);
    }

    fn propagate_call_return(
        &self,
        in_out: &mut BitSet<Local>,
        _call_bb: mir::BasicBlock,
        _dest_bb: mir::BasicBlock,
        dest_place: &mir::Place<'tcx>,
    ) {
        // Calls initialize their destination.
        if let Some(local) = find_local(dest_place) {
            in_out.insert(local);
        }
    }
}

impl<'a, 'tcx> BottomValue for MaybeInitializedLocals<'a, 'tcx> {
    // bottom = definitely uninit
    const BOTTOM_VALUE: bool = false;
}

struct MaybeInitializedLocalsVisitor<'gk> {
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

impl<'tcx> Visitor<'tcx> for MaybeInitializedLocalsVisitor<'_> {
    fn visit_place(
        &mut self,
        place: &Place<'tcx>,
        context: PlaceContext,
        _location: Location,
    ) {
        match context {
            // Only `Operand::Move` and `StorageDead` can deinitialize a local,
            // and only if they act on the whole local, not just some part of it.
            PlaceContext::NonMutatingUse(NonMutatingUseContext::Move)
            | PlaceContext::NonUse(NonUseContext::StorageDead) => match *place {
                Place { base: PlaceBase::Local(local), projection: None } => {
                    self.trans.kill(local)
                }
                _ => {}
            },

            // Handled separately, in `propagate_call_return`.
            PlaceContext::MutatingUse(MutatingUseContext::Call) => {}

            // Any assignment to any part of the local initializes it
            // (even if only partially).
            _ => {
                if context.is_place_assignment() {
                    if let Some(local) = find_local(place) {
                        self.trans.gen(local);
                    }
                }
            }
        }
    }
}
