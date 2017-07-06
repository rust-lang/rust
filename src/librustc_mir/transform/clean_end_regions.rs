// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! This module provides one pass, `CleanEndRegions`, that reduces the
//! set of `EndRegion` statements in the MIR.
//!
//! The "pass" is actually implemented as two traversals (aka visits)
//! of the input MIR. The first traversal, `GatherBorrowedRegions`,
//! finds all of the regions in the MIR that are involved in a borrow.
//!
//! The second traversal, `DeleteTrivialEndRegions`, walks over the
//! MIR and removes any `EndRegion` that is applied to a region that
//! was not seen in the previous pass.

use rustc_data_structures::fx::FxHashSet;

use rustc::middle::region::CodeExtent;
use rustc::mir::transform::{MirPass, MirSource};
use rustc::mir::{BasicBlock, Location, Mir, Rvalue, Statement, StatementKind};
use rustc::mir::visit::{MutVisitor, Visitor};
use rustc::ty::{RegionKind, TyCtxt};

pub struct CleanEndRegions;

struct GatherBorrowedRegions {
    seen_regions: FxHashSet<CodeExtent>,
}

struct DeleteTrivialEndRegions<'a> {
    seen_regions: &'a FxHashSet<CodeExtent>,
}

impl MirPass for CleanEndRegions {
    fn run_pass<'a, 'tcx>(&self,
                          _tcx: TyCtxt<'a, 'tcx, 'tcx>,
                          _source: MirSource,
                          mir: &mut Mir<'tcx>) {
        let mut gather = GatherBorrowedRegions { seen_regions: FxHashSet() };
        gather.visit_mir(mir);

        let mut delete = DeleteTrivialEndRegions { seen_regions: &mut gather.seen_regions };
        delete.visit_mir(mir);
    }
}

impl<'tcx> Visitor<'tcx> for GatherBorrowedRegions {
    fn visit_rvalue(&mut self,
                    rvalue: &Rvalue<'tcx>,
                    location: Location) {
        if let Rvalue::Ref(r, _, _) = *rvalue {
            if let RegionKind::ReScope(ce) = *r {
                self.seen_regions.insert(ce);
            }
        }
        self.super_rvalue(rvalue, location);
    }
}

impl<'a, 'tcx> MutVisitor<'tcx> for DeleteTrivialEndRegions<'a> {
    fn visit_statement(&mut self,
                       block: BasicBlock,
                       statement: &mut Statement<'tcx>,
                       location: Location) {
        let mut delete_it = false;

        if let StatementKind::EndRegion(ref extent) = statement.kind {
            if !self.seen_regions.contains(extent) {
                delete_it = true;
            }
        }

        if delete_it {
            statement.kind = StatementKind::Nop;
        }
        self.super_statement(block, statement, location);
    }
}
