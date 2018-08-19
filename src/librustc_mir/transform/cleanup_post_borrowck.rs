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
//!   - `CleanUserAssertTy`, that replaces all `UserAssertTy` statements
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
//! The `CleanUserAssertTy` pass runs at a distinct time from the
//! `CleanEndRegions` pass. It is important that the `CleanUserAssertTy`
//! pass runs after the MIR borrowck so that the NLL type checker can
//! perform the type assertion when it encounters the `UserAssertTy`
//! statements.

use rustc_data_structures::fx::FxHashSet;

use rustc::middle::region;
use rustc::mir::{BasicBlock, Location, Mir, Rvalue, Statement, StatementKind};
use rustc::mir::visit::{MutVisitor, Visitor, TyContext};
use rustc::ty::{Ty, RegionKind, TyCtxt};
use transform::{MirPass, MirSource};

pub struct CleanEndRegions;

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

        let mut gather = GatherBorrowedRegions {
            seen_regions: FxHashSet()
        };
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
        for re in ty.walk().flat_map(|t| t.regions()) {
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

pub struct CleanUserAssertTy;

pub struct DeleteUserAssertTy;

impl MirPass for CleanUserAssertTy {
    fn run_pass<'a, 'tcx>(&self,
                          _tcx: TyCtxt<'a, 'tcx, 'tcx>,
                          _source: MirSource,
                          mir: &mut Mir<'tcx>) {
        let mut delete = DeleteUserAssertTy;
        delete.visit_mir(mir);
    }
}

impl<'tcx> MutVisitor<'tcx> for DeleteUserAssertTy {
    fn visit_statement(&mut self,
                       block: BasicBlock,
                       statement: &mut Statement<'tcx>,
                       location: Location) {
        if let StatementKind::UserAssertTy(..) = statement.kind {
            statement.make_nop();
        }
        self.super_statement(block, statement, location);
    }
}
