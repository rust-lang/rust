use rustc_middle::mir::Body;
use rustc_middle::ty::TyCtxt;
use rustc_mir_dataflow::rustc_peek::sanity_check;

use crate::MirLint;

pub(super) struct SanityCheck;

impl<'tcx> MirLint<'tcx> for SanityCheck {
    fn run_lint(&self, tcx: TyCtxt<'tcx>, body: &Body<'tcx>) {
        sanity_check(tcx, body);
    }
}
