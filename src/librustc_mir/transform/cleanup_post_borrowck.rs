// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! This module provides two passes:
//!
//!   - [CleanAscribeUserType], that replaces all
//!     [StatementKind::AscribeUserType] statements with [StatementKind::Nop].
//!   - [CleanFakeReadsAndBorrows], that replaces all [FakeRead] statements and
//!     borrows that are read by [FakeReadCause::ForMatchGuard] fake reads with
//!     [StatementKind::Nop].
//!
//! The [CleanFakeReadsAndBorrows] "pass" is actually implemented as two
//! traversals (aka visits) of the input MIR. The first traversal,
//! [DeleteAndRecordFakeReads], deletes the fake reads and finds the temporaries
//! read by [ForMatchGuard] reads, and [DeleteFakeBorrows] deletes the
//! initialization of those temporaries.

use rustc_data_structures::fx::FxHashSet;

use rustc::mir::{BasicBlock, FakeReadCause, Local, Location, Mir, Place};
use rustc::mir::{Statement, StatementKind};
use rustc::mir::visit::MutVisitor;
use rustc::ty::TyCtxt;
use transform::{MirPass, MirSource};

pub struct CleanAscribeUserType;

pub struct DeleteAscribeUserType;

impl MirPass for CleanAscribeUserType {
    fn run_pass<'a, 'tcx>(&self,
                          _tcx: TyCtxt<'a, 'tcx, 'tcx>,
                          _source: MirSource,
                          mir: &mut Mir<'tcx>) {
        let mut delete = DeleteAscribeUserType;
        delete.visit_mir(mir);
    }
}

impl<'tcx> MutVisitor<'tcx> for DeleteAscribeUserType {
    fn visit_statement(&mut self,
                       block: BasicBlock,
                       statement: &mut Statement<'tcx>,
                       location: Location) {
        if let StatementKind::AscribeUserType(..) = statement.kind {
            statement.make_nop();
        }
        self.super_statement(block, statement, location);
    }
}

pub struct CleanFakeReadsAndBorrows;

#[derive(Default)]
pub struct DeleteAndRecordFakeReads {
    fake_borrow_temporaries: FxHashSet<Local>,
}

pub struct DeleteFakeBorrows {
    fake_borrow_temporaries: FxHashSet<Local>,
}

// Removes any FakeReads from the MIR
impl MirPass for CleanFakeReadsAndBorrows {
    fn run_pass<'a, 'tcx>(&self,
                          _tcx: TyCtxt<'a, 'tcx, 'tcx>,
                          _source: MirSource,
                          mir: &mut Mir<'tcx>) {
        let mut delete_reads = DeleteAndRecordFakeReads::default();
        delete_reads.visit_mir(mir);
        let mut delete_borrows = DeleteFakeBorrows {
            fake_borrow_temporaries: delete_reads.fake_borrow_temporaries,
        };
        delete_borrows.visit_mir(mir);
    }
}

impl<'tcx> MutVisitor<'tcx> for DeleteAndRecordFakeReads {
    fn visit_statement(&mut self,
                       block: BasicBlock,
                       statement: &mut Statement<'tcx>,
                       location: Location) {
        if let StatementKind::FakeRead(cause, ref place) = statement.kind {
            if let FakeReadCause::ForMatchGuard = cause {
                match *place {
                    Place::Local(local) => self.fake_borrow_temporaries.insert(local),
                    _ => bug!("Fake match guard read of non-local: {:?}", place),
                };
            }
            statement.make_nop();
        }
        self.super_statement(block, statement, location);
    }
}

impl<'tcx> MutVisitor<'tcx> for DeleteFakeBorrows {
    fn visit_statement(&mut self,
                       block: BasicBlock,
                       statement: &mut Statement<'tcx>,
                       location: Location) {
        if let StatementKind::Assign(Place::Local(local), _) = statement.kind {
            if self.fake_borrow_temporaries.contains(&local) {
                statement.make_nop();
            }
        }
        self.super_statement(block, statement, location);
    }
}
