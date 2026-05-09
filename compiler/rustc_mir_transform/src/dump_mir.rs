//! This pass just dumps MIR at a specified point.

use rustc_middle::mir::Body;
use rustc_middle::ty::TyCtxt;

pub(super) struct Marker(pub &'static str);

impl<'tcx> crate::MirPass<'tcx> for Marker {
    fn name(&self) -> &'static str {
        self.0
    }

    fn run_pass(&self, _tcx: TyCtxt<'tcx>, _body: &mut Body<'tcx>) {}

    fn is_required(&self) -> bool {
        false
    }
}
