//! This pass removes the unwind branch of all the terminators when the no-landing-pads option is
//! specified.

use rustc::ty::TyCtxt;
use rustc::mir::*;
use rustc::mir::visit::MutVisitor;
use crate::transform::{MirPass, MirSource};

pub struct NoLandingPads;

impl MirPass for NoLandingPads {
    fn run_pass<'a, 'tcx>(&self,
                          tcx: TyCtxt<'a, 'tcx, 'tcx>,
                          _: MirSource<'tcx>,
                          mir: &mut Mir<'tcx>) {
        no_landing_pads(tcx, mir)
    }
}

pub fn no_landing_pads<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>, mir: &mut Mir<'tcx>) {
    if tcx.sess.no_landing_pads() {
        NoLandingPads.visit_mir(mir);
    }
}

impl<'tcx> MutVisitor<'tcx> for NoLandingPads {
    fn visit_terminator(&mut self,
                        bb: BasicBlock,
                        terminator: &mut Terminator<'tcx>,
                        location: Location) {
        if let Some(unwind) = terminator.kind.unwind_mut() {
            unwind.take();
        }
        self.super_terminator(bb, terminator, location);
    }
}
