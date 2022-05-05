use super::BackendTypes;
use rustc_hir::def_id::DefId;
use rustc_middle::mir::coverage::*;
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
    /// Returns true if the function source hash was added to the coverage map (even if it had
    /// already been added, for this instance). Returns false *only* if `-C instrument-coverage` is
    /// not enabled (a coverage map is not being generated).
    fn set_function_source_hash(
        &mut self,
        instance: Instance<'tcx>,
        function_source_hash: u64,
    ) -> bool;

    /// Returns true if the counter was added to the coverage map; false if `-C instrument-coverage`
    /// is not enabled (a coverage map is not being generated).
    fn add_coverage_counter(
        &mut self,
        instance: Instance<'tcx>,
        index: CounterValueReference,
        region: CodeRegion,
    ) -> bool;

    /// Returns true if the expression was added to the coverage map; false if
    /// `-C instrument-coverage` is not enabled (a coverage map is not being generated).
    fn add_coverage_counter_expression(
        &mut self,
        instance: Instance<'tcx>,
        id: InjectedExpressionId,
        lhs: ExpressionOperandId,
        op: Op,
        rhs: ExpressionOperandId,
        region: Option<CodeRegion>,
    ) -> bool;

    /// Returns true if the region was added to the coverage map; false if `-C instrument-coverage`
    /// is not enabled (a coverage map is not being generated).
    fn add_coverage_unreachable(&mut self, instance: Instance<'tcx>, region: CodeRegion) -> bool;
}
