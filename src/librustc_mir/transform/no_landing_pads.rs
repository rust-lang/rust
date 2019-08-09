//! This pass removes the unwind branch of all the terminators when the no-landing-pads option is
//! specified.

use rustc::ty::TyCtxt;
use rustc::mir::*;
use rustc::mir::visit::MutVisitor;
use crate::transform::{MirPass, MirSource};

pub struct NoLandingPads;

impl MirPass for NoLandingPads {
    fn run_pass<'tcx>(&self, tcx: TyCtxt<'tcx>, _: MirSource<'tcx>, body: &mut Body<'tcx>) {
        no_landing_pads(tcx, body)
    }
}

pub fn no_landing_pads<'tcx>(tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
    if tcx.sess.no_landing_pads() {
        NoLandingPads.visit_body(body);
    }
}

impl<'tcx> MutVisitor<'tcx> for NoLandingPads {
    fn visit_terminator_kind(&mut self,
                        kind: &mut TerminatorKind<'tcx>,
                        location: Location) {
        if let Some(unwind) = kind.unwind_mut() {
            unwind.take();
        }
        self.super_terminator_kind(kind, location);
    }
}
