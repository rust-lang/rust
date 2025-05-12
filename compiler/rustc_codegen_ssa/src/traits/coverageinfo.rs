use rustc_middle::mir::coverage::CoverageKind;
use rustc_middle::ty::Instance;

pub trait CoverageInfoBuilderMethods<'tcx> {
    /// Performs any start-of-function codegen needed for coverage instrumentation.
    ///
    /// Can be a no-op in backends that don't support coverage instrumentation.
    fn init_coverage(&mut self, _instance: Instance<'tcx>) {}

    /// Handle the MIR coverage info in a backend-specific way.
    ///
    /// This can potentially be a no-op in backends that don't support
    /// coverage instrumentation.
    fn add_coverage(&mut self, instance: Instance<'tcx>, kind: &CoverageKind);
}
