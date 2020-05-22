//! This pass removes the unwind branch of all the terminators when the no-landing-pads option is
//! specified.

use crate::transform::{MirPass, MirSource};
use rustc_middle::mir::visit::MutVisitor;
use rustc_middle::mir::*;
use rustc_middle::ty::TyCtxt;
use rustc_target::spec::PanicStrategy;

pub struct NoLandingPads<'tcx> {
    tcx: TyCtxt<'tcx>,
}

impl<'tcx> NoLandingPads<'tcx> {
    pub fn new(tcx: TyCtxt<'tcx>) -> Self {
        NoLandingPads { tcx }
    }
}

impl<'tcx> MirPass<'tcx> for NoLandingPads<'tcx> {
    fn run_pass(&self, tcx: TyCtxt<'tcx>, _: MirSource<'tcx>, body: &mut Body<'tcx>) {
        no_landing_pads(tcx, body)
    }
}

pub fn no_landing_pads<'tcx>(tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
    if tcx.sess.panic_strategy() == PanicStrategy::Abort {
        NoLandingPads::new(tcx).visit_body(body);
    }
}

impl<'tcx> MutVisitor<'tcx> for NoLandingPads<'tcx> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn visit_terminator_kind(&mut self, kind: &mut TerminatorKind<'tcx>, location: Location) {
        if let Some(unwind) = kind.unwind_mut() {
            unwind.take();
        }
        self.super_terminator_kind(kind, location);
    }
}
