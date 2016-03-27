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
use rustc::mir::repr::*;
use rustc::mir::visit::MutVisitor;
use rustc::mir::transform::{MirPass, Pass};
use syntax::ast::NodeId;

struct EraseRegionsVisitor<'a, 'tcx: 'a> {
    tcx: &'a TyCtxt<'tcx>,
}

impl<'a, 'tcx> EraseRegionsVisitor<'a, 'tcx> {
    pub fn new(tcx: &'a TyCtxt<'tcx>) -> Self {
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
        *substs = self.tcx.mk_substs(self.tcx.erase_regions(*substs));
    }
}

pub struct EraseRegions;

impl Pass for EraseRegions {}

impl<'tcx> MirPass<'tcx> for EraseRegions {
    fn run_pass(&mut self, tcx: &TyCtxt<'tcx>, _: NodeId, mir: &mut Mir<'tcx>) {
        EraseRegionsVisitor::new(tcx).visit_mir(mir);
    }
}
