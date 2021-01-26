//! This pass removes the unwind branch of all the terminators when the no-landing-pads option is
//! specified.

use crate::transform::MirPass;
use rustc_middle::mir::*;
use rustc_middle::ty::TyCtxt;
use rustc_target::spec::PanicStrategy;

pub struct NoLandingPads;

impl<'tcx> MirPass<'tcx> for NoLandingPads {
    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        no_landing_pads(tcx, body)
    }
}

pub fn no_landing_pads<'tcx>(tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
    if tcx.sess.panic_strategy() != PanicStrategy::Abort {
        return;
    }

    for block in body.basic_blocks_mut() {
        let terminator = block.terminator_mut();
        if let Some(unwind) = terminator.kind.unwind_mut() {
            unwind.take();
        }
    }
}
