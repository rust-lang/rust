use rustc_middle::mir::Body;
use rustc_middle::ty::TyCtxt;

/// Dummy pass that does nothing, to be used as the final pass.
///
/// Its existence allows `-Zdump-mir` and mir-opt tests to easily dump
/// the final MIR just before codegen, by dumping the input or output
/// of the `PreCodegen` pass.
pub(super) struct PreCodegen;

impl<'tcx> crate::MirPass<'tcx> for PreCodegen {
    fn run_pass(&self, _tcx: TyCtxt<'tcx>, _body: &mut Body<'tcx>) {
        // This is a dummy pass, so the pass itself doesn't do anything.
        // Dumping is performed by the normal `-Zdump-mir` machinery.
    }

    fn is_required(&self) -> bool {
        false
    }
}
