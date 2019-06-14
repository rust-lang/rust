//! Performs various peephole optimizations.

use rustc::mir::{Constant, Location, Place, PlaceBase, Body, Operand, ProjectionElem, Rvalue,
    Local};
use rustc::mir::visit::{MutVisitor, Visitor};
use rustc::ty::{self, TyCtxt};
use rustc::util::nodemap::{FxHashMap, FxHashSet};
use rustc_data_structures::indexed_vec::Idx;
use std::mem;
use crate::transform::{MirPass, MirSource};

pub struct InstCombine;

impl MirPass for InstCombine {
    fn run_pass<'tcx>(&self, tcx: TyCtxt<'tcx>, _: MirSource<'tcx>, body: &mut Body<'tcx>) {
        // We only run when optimizing MIR (at any level).
        if tcx.sess.opts.debugging_opts.mir_opt_level == 0 {
            return
        }

        // First, find optimization opportunities. This is done in a pre-pass to keep the MIR
        // read-only so that we can do global analyses on the MIR in the process (e.g.
        // `Place::ty()`).
        let optimizations = {
            let mut optimization_finder = OptimizationFinder::new(body, tcx);
            optimization_finder.visit_body(body);
            optimization_finder.optimizations
        };

        // Then carry out those optimizations.
        MutVisitor::visit_body(&mut InstCombineVisitor { optimizations }, body);
    }
}

pub struct InstCombineVisitor<'tcx> {
    optimizations: OptimizationList<'tcx>,
}

impl<'tcx> MutVisitor<'tcx> for InstCombineVisitor<'tcx> {
    fn visit_rvalue(&mut self, rvalue: &mut Rvalue<'tcx>, location: Location) {
        if self.optimizations.and_stars.remove(&location) {
            debug!("Replacing `&*`: {:?}", rvalue);
            let new_place = match *rvalue {
                Rvalue::Ref(_, _, Place::Projection(ref mut projection)) => {
                    // Replace with dummy
                    mem::replace(&mut projection.base, Place::Base(PlaceBase::Local(Local::new(0))))
                }
                _ => bug!("Detected `&*` but didn't find `&*`!"),
            };
            *rvalue = Rvalue::Use(Operand::Copy(new_place))
        }

        if let Some(constant) = self.optimizations.arrays_lengths.remove(&location) {
            debug!("Replacing `Len([_; N])`: {:?}", rvalue);
            *rvalue = Rvalue::Use(Operand::Constant(box constant));
        }

        self.super_rvalue(rvalue, location)
    }
}

/// Finds optimization opportunities on the MIR.
struct OptimizationFinder<'b, 'tcx> {
    body: &'b Body<'tcx>,
    tcx: TyCtxt<'tcx>,
    optimizations: OptimizationList<'tcx>,
}

impl OptimizationFinder<'b, 'tcx> {
    fn new(body: &'b Body<'tcx>, tcx: TyCtxt<'tcx>) -> OptimizationFinder<'b, 'tcx> {
        OptimizationFinder {
            body,
            tcx,
            optimizations: OptimizationList::default(),
        }
    }
}

impl Visitor<'tcx> for OptimizationFinder<'b, 'tcx> {
    fn visit_rvalue(&mut self, rvalue: &Rvalue<'tcx>, location: Location) {
        if let Rvalue::Ref(_, _, Place::Projection(ref projection)) = *rvalue {
            if let ProjectionElem::Deref = projection.elem {
                if projection.base.ty(self.body, self.tcx).ty.is_region_ptr() {
                    self.optimizations.and_stars.insert(location);
                }
            }
        }

        if let Rvalue::Len(ref place) = *rvalue {
            let place_ty = place.ty(&self.body.local_decls, self.tcx).ty;
            if let ty::Array(_, len) = place_ty.sty {
                let span = self.body.source_info(location).span;
                let ty = self.tcx.types.usize;
                let constant = Constant { span, ty, literal: len, user_ty: None };
                self.optimizations.arrays_lengths.insert(location, constant);
            }
        }

        self.super_rvalue(rvalue, location)
    }
}

#[derive(Default)]
struct OptimizationList<'tcx> {
    and_stars: FxHashSet<Location>,
    arrays_lengths: FxHashMap<Location, Constant<'tcx>>,
}
