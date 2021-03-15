use super::BackendTypes;
use rustc_hir::def_id::DefId;
use rustc_middle::mir::coverage::*;
use rustc_middle::ty::Instance;

pub trait CoverageInfoMethods<'tcx>: BackendTypes {
    fn coverageinfo_finalize(&self);

    /// Functions with MIR-based coverage are normally codegenned _only_ if
    /// called. LLVM coverage tools typically expect every function to be
    /// defined (even if unused), with at least one call to LLVM intrinsic
    /// `instrprof.increment`.
    ///
    /// Codegen a small function that will never be called, with one counter
    /// that will never be incremented.
    ///
    /// For used/called functions, the coverageinfo was already added to the
    /// `function_coverage_map` (keyed by function `Instance`) during codegen.
    /// But in this case, since the unused function was _not_ previously
    /// codegenned, collect the coverage `CodeRegion`s from the MIR and add
    /// them. The first `CodeRegion` is used to add a single counter, with the
    /// same counter ID used in the injected `instrprof.increment` intrinsic
    /// call. Since the function is never called, all other `CodeRegion`s can be
    /// added as `unreachable_region`s.
    fn define_unused_fn(&self, def_id: DefId);

    /// For LLVM codegen, returns a function-specific `Value` for a global
    /// string, to hold the function name passed to LLVM intrinsic
    /// `instrprof.increment()`. The `Value` is only created once per instance.
    /// Multiple invocations with the same instance return the same `Value`.
    fn get_pgo_func_name_var(&self, instance: Instance<'tcx>) -> Self::Value;

    /// Creates a new PGO function name variable. This should only be called
    /// to fill in the unused function names array.
    fn create_pgo_func_name_var(&self, instance: Instance<'tcx>) -> Self::Value;
}

pub trait CoverageInfoBuilderMethods<'tcx>: BackendTypes {
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
