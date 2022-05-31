//! This module implements a dead store elimination (DSE) routine.
//!
//! This transformation was written specifically for the needs of dest prop. Although it is
//! perfectly sound to use it in any context that might need it, its behavior should not be changed
//! without analyzing the interaction this will have with dest prop. Specifically, in addition to
//! the soundness of this pass in general, dest prop needs it to satisfy two additional conditions:
//!
//!  1. It's idempotent, meaning that running this pass a second time immediately after running it a
//!     first time will not cause any further changes.
//!  2. This idempotence persists across dest prop's main transform, in other words inserting any
//!     number of iterations of dest prop between the first and second application of this transform
//!     will still not cause any further changes.
//!

use rustc_index::bit_set::BitSet;
use rustc_middle::{
    mir::{visit::Visitor, *},
    ty::TyCtxt,
};
use rustc_mir_dataflow::{impls::MaybeTransitiveLiveLocals, Analysis};

/// Performs the optimization on the body
///
/// The `borrowed` set must be a `BitSet` of all the locals that are ever borrowed in this body. It
/// can be generated via the [`get_borrowed_locals`] function.
pub fn eliminate<'tcx>(tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>, borrowed: &BitSet<Local>) {
    let mut live = MaybeTransitiveLiveLocals::new(borrowed)
        .into_engine(tcx, body)
        .iterate_to_fixpoint()
        .into_results_cursor(body);

    let mut patch = Vec::new();
    for (bb, bb_data) in traversal::preorder(body) {
        for (statement_index, statement) in bb_data.statements.iter().enumerate().rev() {
            let loc = Location { block: bb, statement_index };
            if let StatementKind::Assign(assign) = &statement.kind {
                if assign.1.is_pointer_int_cast() {
                    continue;
                }
            }
            match &statement.kind {
                StatementKind::Assign(box (place, _))
                | StatementKind::SetDiscriminant { place: box place, .. }
                | StatementKind::Deinit(box place) => {
                    if !place.is_indirect() && !borrowed.contains(place.local) {
                        live.seek_before_primary_effect(loc);
                        if !live.get().contains(place.local) {
                            patch.push(loc);
                        }
                    }
                }
                StatementKind::Retag(_, _)
                | StatementKind::StorageLive(_)
                | StatementKind::StorageDead(_)
                | StatementKind::Coverage(_)
                | StatementKind::CopyNonOverlapping(_)
                | StatementKind::Nop => (),

                StatementKind::FakeRead(_) | StatementKind::AscribeUserType(_, _) => {
                    bug!("{:?} not found in this MIR phase!", &statement.kind)
                }
            }
        }
    }

    if patch.is_empty() {
        return;
    }

    let bbs = body.basic_blocks_mut();
    for Location { block, statement_index } in patch {
        bbs[block].statements[statement_index].make_nop();
    }
}

pub fn get_borrowed_locals(body: &Body<'_>) -> BitSet<Local> {
    let mut b = BorrowedLocals(BitSet::new_empty(body.local_decls.len()));
    b.visit_body(body);
    b.0
}

struct BorrowedLocals(BitSet<Local>);

impl<'tcx> Visitor<'tcx> for BorrowedLocals {
    fn visit_rvalue(&mut self, rvalue: &Rvalue<'tcx>, loc: Location) {
        self.super_rvalue(rvalue, loc);
        match rvalue {
            Rvalue::AddressOf(_, borrowed_place) | Rvalue::Ref(_, _, borrowed_place) => {
                if !borrowed_place.is_indirect() {
                    self.0.insert(borrowed_place.local);
                }
            }

            Rvalue::Cast(..)
            | Rvalue::ShallowInitBox(..)
            | Rvalue::Use(..)
            | Rvalue::Repeat(..)
            | Rvalue::Len(..)
            | Rvalue::BinaryOp(..)
            | Rvalue::CheckedBinaryOp(..)
            | Rvalue::NullaryOp(..)
            | Rvalue::UnaryOp(..)
            | Rvalue::Discriminant(..)
            | Rvalue::Aggregate(..)
            | Rvalue::ThreadLocalRef(..) => {}
        }
    }

    fn visit_terminator(&mut self, terminator: &Terminator<'tcx>, location: Location) {
        self.super_terminator(terminator, location);

        match terminator.kind {
            TerminatorKind::Drop { place: dropped_place, .. } => {
                if !dropped_place.is_indirect() {
                    self.0.insert(dropped_place.local);
                }
            }

            TerminatorKind::Abort
            | TerminatorKind::DropAndReplace { .. }
            | TerminatorKind::Assert { .. }
            | TerminatorKind::Call { .. }
            | TerminatorKind::FalseEdge { .. }
            | TerminatorKind::FalseUnwind { .. }
            | TerminatorKind::GeneratorDrop
            | TerminatorKind::Goto { .. }
            | TerminatorKind::Resume
            | TerminatorKind::Return
            | TerminatorKind::SwitchInt { .. }
            | TerminatorKind::Unreachable
            | TerminatorKind::Yield { .. }
            | TerminatorKind::InlineAsm { .. } => {}
        }
    }
}

pub struct DeadStoreElimination;

impl<'tcx> MirPass<'tcx> for DeadStoreElimination {
    fn is_enabled(&self, sess: &rustc_session::Session) -> bool {
        sess.mir_opt_level() >= 2
    }

    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        let borrowed = get_borrowed_locals(body);
        eliminate(tcx, body, &borrowed);
    }
}
