use super::BackendTypes;
use rustc_middle::mir::Coverage;
use rustc_middle::ty::Instance;

pub trait CoverageInfoBuilderMethods<'tcx>: BackendTypes {
    /// Handle the MIR coverage info in a backend-specific way.
    ///
    /// This can potentially be a no-op in backends that don't support
    /// coverage instrumentation.
    fn add_coverage(&mut self, instance: Instance<'tcx>, coverage: &Coverage);
}
