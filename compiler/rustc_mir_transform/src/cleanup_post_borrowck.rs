//! This module provides a pass to replacing the following statements with
//! [`Nop`]s
//!
//!   - [`AscribeUserType`]
//!   - [`FakeRead`]
//!   - [`Assign`] statements with a [`Shallow`] borrow
//!
//! The `CleanFakeReadsAndBorrows` "pass" is actually implemented as two
//! traversals (aka visits) of the input MIR. The first traversal,
//! `DeleteAndRecordFakeReads`, deletes the fake reads and finds the
//! temporaries read by [`ForMatchGuard`] reads, and `DeleteFakeBorrows`
//! deletes the initialization of those temporaries.
//!
//! [`AscribeUserType`]: rustc_middle::mir::StatementKind::AscribeUserType
//! [`Shallow`]: rustc_middle::mir::BorrowKind::Shallow
//! [`FakeRead`]: rustc_middle::mir::StatementKind::FakeRead
//! [`Assign`]: rustc_middle::mir::StatementKind::Assign
//! [`ForMatchGuard`]: rustc_middle::mir::FakeReadCause::ForMatchGuard
//! [`Nop`]: rustc_middle::mir::StatementKind::Nop

use crate::MirPass;
use rustc_middle::mir::{Body, BorrowKind, Rvalue, StatementKind};
use rustc_middle::ty::TyCtxt;

pub struct CleanupNonCodegenStatements;

impl<'tcx> MirPass<'tcx> for CleanupNonCodegenStatements {
    fn run_pass(&self, _tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        for basic_block in body.basic_blocks.as_mut_preserves_cfg() {
            for statement in basic_block.statements.iter_mut() {
                match statement.kind {
                    StatementKind::AscribeUserType(..)
                    | StatementKind::Assign(box (_, Rvalue::Ref(_, BorrowKind::Shallow, _)))
                    | StatementKind::FakeRead(..) => statement.make_nop(),
                    _ => (),
                }
            }
        }

        body.user_type_annotations.raw.clear();

        for decl in &mut body.local_decls {
            decl.user_ty = None;
        }
    }
}
