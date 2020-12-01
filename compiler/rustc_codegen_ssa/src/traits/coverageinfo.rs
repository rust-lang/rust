use super::BackendTypes;
use rustc_middle::mir::coverage::*;
use rustc_middle::ty::Instance;

pub trait CoverageInfoMethods: BackendTypes {
    fn coverageinfo_finalize(&self);
}

pub trait CoverageInfoBuilderMethods<'tcx>: BackendTypes {
    fn create_pgo_func_name_var(&self, instance: Instance<'tcx>) -> Self::Value;

    /// Returns true if the function source hash was added to the coverage map (even if it had
    /// already been added, for this instance). Returns false *only* if `-Z instrument-coverage` is
    /// not enabled (a coverage map is not being generated).
    fn set_function_source_hash(
        &mut self,
        instance: Instance<'tcx>,
        function_source_hash: u64,
    ) -> bool;

    /// Returns true if the counter was added to the coverage map; false if `-Z instrument-coverage`
    /// is not enabled (a coverage map is not being generated).
    fn add_coverage_counter(
        &mut self,
        instance: Instance<'tcx>,
        index: CounterValueReference,
        region: CodeRegion,
    ) -> bool;

    /// Returns true if the expression was added to the coverage map; false if
    /// `-Z instrument-coverage` is not enabled (a coverage map is not being generated).
    fn add_coverage_counter_expression(
        &mut self,
        instance: Instance<'tcx>,
        id: InjectedExpressionId,
        lhs: ExpressionOperandId,
        op: Op,
        rhs: ExpressionOperandId,
        region: Option<CodeRegion>,
    ) -> bool;

    /// Returns true if the region was added to the coverage map; false if `-Z instrument-coverage`
    /// is not enabled (a coverage map is not being generated).
    fn add_coverage_unreachable(&mut self, instance: Instance<'tcx>, region: CodeRegion) -> bool;
}
