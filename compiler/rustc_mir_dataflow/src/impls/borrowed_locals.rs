use rustc_index::bit_set::DenseBitSet;
use rustc_middle::mir::*;

use crate::{Analysis, GenKill};

/// A dataflow analysis that tracks whether a pointer or reference could possibly exist that points
/// to a given local. This analysis ignores fake borrows, so it should not be used by
/// borrowck.
///
/// At present, this is used as a very limited form of alias analysis. For example,
/// `MaybeBorrowedLocals` is used to compute which locals are live during a yield expression for
/// immovable coroutines.
pub struct MaybeBorrowedLocals;

impl MaybeBorrowedLocals {
    pub(super) fn gen_statement(state: &mut DenseBitSet<Local>, stmt: &Statement<'_>) {
        if let StatementKind::Assign((_, rvalue)) = &stmt.kind {
            match rvalue {
                // We ignore fake borrows as these get removed after analysis and shouldn't effect
                // the layout of generators.
                Rvalue::RawPtr(_, borrowed_place)
                | Rvalue::Ref(_, BorrowKind::Mut { .. } | BorrowKind::Shared, borrowed_place)
                | Rvalue::Reborrow(_, _, borrowed_place) => {
                    if !borrowed_place.is_indirect() {
                        state.insert(borrowed_place.local);
                    }
                }

                Rvalue::Cast(..)
                | Rvalue::Ref(_, BorrowKind::Fake(_), _)
                | Rvalue::Use(..)
                | Rvalue::ThreadLocalRef(..)
                | Rvalue::Repeat(..)
                | Rvalue::BinaryOp(..)
                | Rvalue::UnaryOp(..)
                | Rvalue::Discriminant(..)
                | Rvalue::Aggregate(..)
                | Rvalue::CopyForDeref(..)
                | Rvalue::WrapUnsafeBinder(..) => {}
            }
        }
    }

    pub(super) fn gen_terminator(state: &mut DenseBitSet<Local>, terminator: &Terminator<'_>) {
        match terminator.kind {
            TerminatorKind::Drop { place: dropped_place, .. } => {
                // Drop terminators may call custom drop glue (`Drop::drop`), which takes `&mut
                // self` as a parameter. In the general case, a drop impl could launder that
                // reference into the surrounding environment through a raw pointer, thus creating
                // a valid `*mut` pointing to the dropped local. We are not yet willing to declare
                // this particular case UB, so we must treat all dropped locals as mutably borrowed
                // for now. See discussion on [#61069].
                //
                // [#61069]: https://github.com/rust-lang/rust/pull/61069
                if !dropped_place.is_indirect() {
                    state.insert(dropped_place.local);
                }
            }

            TerminatorKind::UnwindTerminate(_)
            | TerminatorKind::Assert { .. }
            | TerminatorKind::Call { .. }
            | TerminatorKind::FalseEdge { .. }
            | TerminatorKind::FalseUnwind { .. }
            | TerminatorKind::CoroutineDrop
            | TerminatorKind::Goto { .. }
            | TerminatorKind::InlineAsm { .. }
            | TerminatorKind::UnwindResume
            | TerminatorKind::Return
            | TerminatorKind::TailCall { .. }
            | TerminatorKind::SwitchInt { .. }
            | TerminatorKind::Unreachable
            | TerminatorKind::Yield { .. } => {}
        }
    }
}

impl<'tcx> Analysis<'tcx> for MaybeBorrowedLocals {
    type Domain = DenseBitSet<Local>;
    const NAME: &'static str = "maybe_borrowed_locals";

    fn bottom_value(&self, body: &Body<'tcx>) -> Self::Domain {
        // bottom = unborrowed
        DenseBitSet::new_empty(body.local_decls().len())
    }

    fn initialize_start_block(&self, _: &Body<'tcx>, _: &mut Self::Domain) {
        // No locals are aliased on function entry
    }

    fn apply_primary_statement_effect(
        &self,
        state: &mut Self::Domain,
        statement: &Statement<'tcx>,
        _location: Location,
    ) {
        Self::gen_statement(state, statement);

        // When we reach a `StorageDead` statement, we can assume that any pointers to this memory
        // are now invalid.
        if let StatementKind::StorageDead(local) = statement.kind {
            state.kill(local);
        }
    }

    fn apply_primary_terminator_effect(
        &self,
        state: &mut Self::Domain,
        terminator: &Terminator<'tcx>,
        _location: Location,
    ) {
        Self::gen_terminator(state, terminator);
    }
}

/// The set of locals that are borrowed at some point in the MIR body.
pub fn borrowed_locals(body: &Body<'_>) -> DenseBitSet<Local> {
    let mut borrowed = DenseBitSet::new_empty(body.local_decls.len());
    for bb_data in body.basic_blocks.iter() {
        for stmt in &bb_data.statements {
            MaybeBorrowedLocals::gen_statement(&mut borrowed, stmt);
        }
        MaybeBorrowedLocals::gen_terminator(&mut borrowed, bb_data.terminator());
    }
    borrowed
}
