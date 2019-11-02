use rustc::mir::visit::Visitor;
use rustc::mir::{self, Local, Location};
use rustc::ty::{self, TyCtxt};
use rustc_index::bit_set::BitSet;
use syntax_pos::DUMMY_SP;

use crate::dataflow::{self, GenKillSet};

/// Whether a borrow to a `Local` has been created that could allow that `Local` to be mutated
/// indirectly. This could either be a mutable reference (`&mut`) or a shared borrow if the type of
/// that `Local` allows interior mutability. Operations that can mutate local's indirectly include:
/// assignments through a pointer (`*p = 42`), function calls, drop terminators and inline assembly.
///
/// If this returns false for a `Local` at a given statement (or terminator), that `Local` could
/// not possibly have been mutated indirectly prior to that statement.
#[derive(Copy, Clone)]
pub struct IndirectlyMutableLocals<'mir, 'tcx> {
    body: &'mir mir::Body<'tcx>,
    tcx: TyCtxt<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
}

impl<'mir, 'tcx> IndirectlyMutableLocals<'mir, 'tcx> {
    pub fn new(
        tcx: TyCtxt<'tcx>,
        body: &'mir mir::Body<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
    ) -> Self {
        IndirectlyMutableLocals { body, tcx, param_env }
    }

    fn transfer_function<'a>(
        &self,
        trans: &'a mut GenKillSet<Local>,
    ) -> TransferFunction<'a, 'mir, 'tcx> {
        TransferFunction {
            body: self.body,
            tcx: self.tcx,
            param_env: self.param_env,
            trans
        }
    }
}

impl<'mir, 'tcx> dataflow::BitDenotation<'tcx> for IndirectlyMutableLocals<'mir, 'tcx> {
    type Idx = Local;

    fn name() -> &'static str { "mut_borrowed_locals" }

    fn bits_per_block(&self) -> usize {
        self.body.local_decls.len()
    }

    fn start_block_effect(&self, _entry_set: &mut BitSet<Local>) {
        // Nothing is borrowed on function entry
    }

    fn statement_effect(
        &self,
        trans: &mut GenKillSet<Local>,
        loc: Location,
    ) {
        let stmt = &self.body[loc.block].statements[loc.statement_index];
        self.transfer_function(trans).visit_statement(stmt, loc);
    }

    fn terminator_effect(
        &self,
        trans: &mut GenKillSet<Local>,
        loc: Location,
    ) {
        let terminator = self.body[loc.block].terminator();
        self.transfer_function(trans).visit_terminator(terminator, loc);
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

impl<'mir, 'tcx> dataflow::BottomValue for IndirectlyMutableLocals<'mir, 'tcx> {
    // bottom = unborrowed
    const BOTTOM_VALUE: bool = false;
}

/// A `Visitor` that defines the transfer function for `IndirectlyMutableLocals`.
struct TransferFunction<'a, 'mir, 'tcx> {
    trans: &'a mut GenKillSet<Local>,
    body: &'mir mir::Body<'tcx>,
    tcx: TyCtxt<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
}

impl<'tcx> TransferFunction<'_, '_, 'tcx> {
    /// Returns `true` if this borrow would allow mutation of the `borrowed_place`.
    fn borrow_allows_mutation(
        &self,
        kind: mir::BorrowKind,
        borrowed_place: &mir::Place<'tcx>,
    ) -> bool {
        match kind {
            mir::BorrowKind::Mut { .. } => true,

            | mir::BorrowKind::Shared
            | mir::BorrowKind::Shallow
            | mir::BorrowKind::Unique
            => !borrowed_place
                .ty(self.body, self.tcx)
                .ty
                .is_freeze(self.tcx, self.param_env, DUMMY_SP),
        }
    }
}

impl<'tcx> Visitor<'tcx> for TransferFunction<'_, '_, 'tcx> {
    fn visit_rvalue(
        &mut self,
        rvalue: &mir::Rvalue<'tcx>,
        location: Location,
    ) {
        if let mir::Rvalue::Ref(_, kind, ref borrowed_place) = *rvalue {
            if self.borrow_allows_mutation(kind, borrowed_place) {
                match borrowed_place.base {
                    mir::PlaceBase::Local(borrowed_local) if !borrowed_place.is_indirect()
                        => self.trans.gen(borrowed_local),

                    _ => (),
                }
            }
        }

        self.super_rvalue(rvalue, location);
    }


    fn visit_terminator(&mut self, terminator: &mir::Terminator<'tcx>, location: Location) {
        // This method purposely does nothing except call `super_terminator`. It exists solely to
        // document the subtleties around drop terminators.

        self.super_terminator(terminator, location);

        if let mir::TerminatorKind::Drop { location: _, .. }
             | mir::TerminatorKind::DropAndReplace { location: _, .. } = &terminator.kind
        {
            // Although drop terminators mutably borrow the location being dropped, that borrow
            // cannot live beyond the drop terminator because the dropped location is invalidated.
        }
    }
}
