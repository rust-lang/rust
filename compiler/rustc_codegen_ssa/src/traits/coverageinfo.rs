use super::BackendTypes;
use rustc_hir::def_id::DefId;
use rustc_middle::mir::Coverage;
use rustc_middle::ty::Instance;

pub trait CoverageInfoMethods<'tcx>: BackendTypes {
    fn coverageinfo_finalize(&self);

    /// Codegen a small function that will never be called, with one counter
    /// that will never be incremented, that gives LLVM coverage tools a
    /// function definition it needs in order to resolve coverage map references
    /// to unused functions. This is necessary so unused functions will appear
    /// as uncovered (coverage execution count `0`) in LLVM coverage reports.
    fn define_unused_fn(&self, def_id: DefId);

    /// For LLVM codegen, returns a function-specific `Value` for a global
    /// string, to hold the function name passed to LLVM intrinsic
    /// `instrprof.increment()`. The `Value` is only created once per instance.
    /// Multiple invocations with the same instance return the same `Value`.
    fn get_pgo_func_name_var(&self, instance: Instance<'tcx>) -> Self::Value;
}

pub trait CoverageInfoBuilderMethods<'tcx>: BackendTypes {
    /// Handle the MIR coverage info in a backend-specific way.
    ///
    /// This can potentially be a no-op in backends that don't support
    /// coverage instrumentation.
    fn add_coverage(&mut self, instance: Instance<'tcx>, coverage: &Coverage);
}
