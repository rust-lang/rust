// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Performs various peephole optimizations.

use rustc::mir::{Location, Lvalue, Mir, Operand, ProjectionElem, Rvalue, Local};
use rustc::mir::transform::{MirPass, MirSource, Pass};
use rustc::mir::visit::{MutVisitor, Visitor};
use rustc::ty::TyCtxt;
use rustc::util::nodemap::FxHashSet;
use rustc_data_structures::indexed_vec::Idx;
use std::mem;

pub struct InstCombine {
    optimizations: OptimizationList,
}

impl InstCombine {
    pub fn new() -> InstCombine {
        InstCombine {
            optimizations: OptimizationList::default(),
        }
    }
}

impl Pass for InstCombine {}

impl<'tcx> MirPass<'tcx> for InstCombine {
    fn run_pass<'a>(&mut self,
                    tcx: TyCtxt<'a, 'tcx, 'tcx>,
                    _: MirSource,
                    mir: &mut Mir<'tcx>) {
        // We only run when optimizing MIR (at any level).
        if tcx.sess.opts.debugging_opts.mir_opt_level == 0 {
            return
        }

        // First, find optimization opportunities. This is done in a pre-pass to keep the MIR
        // read-only so that we can do global analyses on the MIR in the process (e.g.
        // `Lvalue::ty()`).
        {
            let mut optimization_finder = OptimizationFinder::new(mir, tcx);
            optimization_finder.visit_mir(mir);
            self.optimizations = optimization_finder.optimizations
        }

        // Then carry out those optimizations.
        MutVisitor::visit_mir(&mut *self, mir);
    }
}

impl<'tcx> MutVisitor<'tcx> for InstCombine {
    fn visit_rvalue(&mut self, rvalue: &mut Rvalue<'tcx>, location: Location) {
        if self.optimizations.and_stars.remove(&location) {
            debug!("Replacing `&*`: {:?}", rvalue);
            let new_lvalue = match *rvalue {
                Rvalue::Ref(_, _, Lvalue::Projection(ref mut projection)) => {
                    // Replace with dummy
                    mem::replace(&mut projection.base, Lvalue::Local(Local::new(0)))
                }
                _ => bug!("Detected `&*` but didn't find `&*`!"),
            };
            *rvalue = Rvalue::Use(Operand::Consume(new_lvalue))
        }

        self.super_rvalue(rvalue, location)
    }
}

/// Finds optimization opportunities on the MIR.
struct OptimizationFinder<'b, 'a, 'tcx:'a+'b> {
    mir: &'b Mir<'tcx>,
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    optimizations: OptimizationList,
}

impl<'b, 'a, 'tcx:'b> OptimizationFinder<'b, 'a, 'tcx> {
    fn new(mir: &'b Mir<'tcx>, tcx: TyCtxt<'a, 'tcx, 'tcx>) -> OptimizationFinder<'b, 'a, 'tcx> {
        OptimizationFinder {
            mir: mir,
            tcx: tcx,
            optimizations: OptimizationList::default(),
        }
    }
}

impl<'b, 'a, 'tcx> Visitor<'tcx> for OptimizationFinder<'b, 'a, 'tcx> {
    fn visit_rvalue(&mut self, rvalue: &Rvalue<'tcx>, location: Location) {
        if let Rvalue::Ref(_, _, Lvalue::Projection(ref projection)) = *rvalue {
            if let ProjectionElem::Deref = projection.elem {
                if projection.base.ty(self.mir, self.tcx).to_ty(self.tcx).is_region_ptr() {
                    self.optimizations.and_stars.insert(location);
                }
            }
        }

        self.super_rvalue(rvalue, location)
    }
}

#[derive(Default)]
struct OptimizationList {
    and_stars: FxHashSet<Location>,
}
