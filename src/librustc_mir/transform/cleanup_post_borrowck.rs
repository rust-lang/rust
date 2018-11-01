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
//!   - `CleanEndRegions`, that reduces the set of `EndRegion` statements
//!     in the MIR.
//!   - `CleanAscribeUserType`, that replaces all `AscribeUserType` statements
//!     with `Nop`.
//!
//! The `CleanEndRegions` "pass" is actually implemented as two
//! traversals (aka visits) of the input MIR. The first traversal,
//! `GatherBorrowedRegions`, finds all of the regions in the MIR
//! that are involved in a borrow.
//!
//! The second traversal, `DeleteTrivialEndRegions`, walks over the
//! MIR and removes any `EndRegion` that is applied to a region that
//! was not seen in the previous pass.
//!
//! The `CleanAscribeUserType` pass runs at a distinct time from the
//! `CleanEndRegions` pass. It is important that the `CleanAscribeUserType`
//! pass runs after the MIR borrowck so that the NLL type checker can
//! perform the type assertion when it encounters the `AscribeUserType`
//! statements.

use rustc_data_structures::fx::FxHashSet;

use rustc::middle::region;
use rustc::mir::{BasicBlock, FakeReadCause, Local, Location, Mir, Place};
use rustc::mir::{Rvalue, Statement, StatementKind};
use rustc::mir::visit::{MutVisitor, Visitor, TyContext};
use rustc::ty::{Ty, RegionKind, TyCtxt};
use smallvec::smallvec;
use transform::{MirPass, MirSource};

pub struct CleanEndRegions;

#[derive(Default)]
struct GatherBorrowedRegions {
    seen_regions: FxHashSet<region::Scope>,
}

struct DeleteTrivialEndRegions<'a> {
    seen_regions: &'a FxHashSet<region::Scope>,
}

impl MirPass for CleanEndRegions {
    fn run_pass<'a, 'tcx>(&self,
                          tcx: TyCtxt<'a, 'tcx, 'tcx>,
                          _source: MirSource,
                          mir: &mut Mir<'tcx>) {
        if !tcx.emit_end_regions() { return; }

        let mut gather = GatherBorrowedRegions::default();
        gather.visit_mir(mir);

        let mut delete = DeleteTrivialEndRegions { seen_regions: &mut gather.seen_regions };
        delete.visit_mir(mir);
    }
}

impl<'tcx> Visitor<'tcx> for GatherBorrowedRegions {
    fn visit_rvalue(&mut self,
                    rvalue: &Rvalue<'tcx>,
                    location: Location) {
        // Gather regions that are used for borrows
        if let Rvalue::Ref(r, _, _) = *rvalue {
            if let RegionKind::ReScope(ce) = *r {
                self.seen_regions.insert(ce);
            }
        }
        self.super_rvalue(rvalue, location);
    }

    fn visit_ty(&mut self, ty: &Ty<'tcx>, _: TyContext) {
        // Gather regions that occur in types
        let mut regions = smallvec![];
        for t in ty.walk() {
            t.push_regions(&mut regions);
        }
        for re in regions {
            match *re {
                RegionKind::ReScope(ce) => { self.seen_regions.insert(ce); }
                _ => {},
            }
        }
        self.super_ty(ty);
    }
}

impl<'a, 'tcx> MutVisitor<'tcx> for DeleteTrivialEndRegions<'a> {
    fn visit_statement(&mut self,
                       block: BasicBlock,
                       statement: &mut Statement<'tcx>,
                       location: Location) {
        let mut delete_it = false;

        if let StatementKind::EndRegion(ref region_scope) = statement.kind {
            if !self.seen_regions.contains(region_scope) {
                delete_it = true;
            }
        }

        if delete_it {
            statement.make_nop();
        }
        self.super_statement(block, statement, location);
    }
}

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
