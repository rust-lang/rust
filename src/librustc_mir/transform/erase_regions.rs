// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! This pass erases all early-bound regions from the types occuring in the MIR.
//! We want to do this once just before trans, so trans does not have to take
//! care erasing regions all over the place.

use rustc::ty::subst::Substs;
use rustc::ty::{Ty, TyCtxt};
use rustc::mir::*;
use rustc::mir::visit::MutVisitor;
use rustc::mir::transform::{MirPass, MirSource, Pass};

struct EraseRegionsVisitor<'a, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
}

impl<'a, 'tcx> EraseRegionsVisitor<'a, 'tcx> {
    pub fn new(tcx: TyCtxt<'a, 'tcx, 'tcx>) -> Self {
        EraseRegionsVisitor {
            tcx: tcx
        }
    }
}

impl<'a, 'tcx> MutVisitor<'tcx> for EraseRegionsVisitor<'a, 'tcx> {
    fn visit_ty(&mut self, ty: &mut Ty<'tcx>) {
        let old_ty = *ty;
        *ty = self.tcx.erase_regions(&old_ty);
    }

    fn visit_substs(&mut self, substs: &mut &'tcx Substs<'tcx>) {
        *substs = self.tcx.erase_regions(&{*substs});
    }
}

pub struct EraseRegions;

impl Pass for EraseRegions {}

impl<'tcx> MirPass<'tcx> for EraseRegions {
    fn run_pass<'a>(&mut self, tcx: TyCtxt<'a, 'tcx, 'tcx>,
                    _: MirSource, mir: &mut Mir<'tcx>) {
        EraseRegionsVisitor::new(tcx).visit_mir(mir);
    }
}
