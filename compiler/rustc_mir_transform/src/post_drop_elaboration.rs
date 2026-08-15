use rustc_const_eval::check_consts;
use rustc_middle::mir::*;
use rustc_middle::ty::TyCtxt;

use crate::MirLint;

pub(super) struct CheckLiveDrops;

impl<'tcx> MirLint<'tcx> for CheckLiveDrops {
    fn run_lint(&self, tcx: TyCtxt<'tcx>, body: &Body<'tcx>) {
        check_consts::post_drop_elaboration::check_live_drops(tcx, body);
    }
}
