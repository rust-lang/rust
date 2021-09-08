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
use rustc_middle::mir::visit::MutVisitor;
use rustc_middle::mir::{Body, BorrowKind, Location, Rvalue};
use rustc_middle::mir::{Statement, StatementKind};
use rustc_middle::ty::TyCtxt;

pub struct CleanupNonCodegenStatements;

pub struct DeleteNonCodegenStatements<'tcx> {
    tcx: TyCtxt<'tcx>,
}

impl<'tcx> MirPass<'tcx> for CleanupNonCodegenStatements {
    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        let mut delete = DeleteNonCodegenStatements { tcx };
        delete.visit_body(body);
        body.user_type_annotations.raw.clear();

        for decl in &mut body.local_decls {
            decl.user_ty = None;
        }
    }
}

impl<'tcx> MutVisitor<'tcx> for DeleteNonCodegenStatements<'tcx> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn visit_statement(&mut self, statement: &mut Statement<'tcx>, location: Location) {
        match statement.kind {
            StatementKind::AscribeUserType(..)
            | StatementKind::Assign(box (_, Rvalue::Ref(_, BorrowKind::Shallow, _)))
            | StatementKind::FakeRead(..) => statement.make_nop(),
            _ => (),
        }
        self.super_statement(statement, location);
    }
}
