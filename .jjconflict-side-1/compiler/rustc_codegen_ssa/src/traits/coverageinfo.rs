use rustc_middle::mir::coverage::CoverageKind;
use rustc_middle::ty::Instance;

pub trait CoverageInfoBuilderMethods<'tcx> {
    /// Handle the MIR coverage info in a backend-specific way.
    ///
    /// This can potentially be a no-op in backends that don't support
    /// coverage instrumentation.
    fn add_coverage(&mut self, instance: Instance<'tcx>, kind: &CoverageKind);
}
