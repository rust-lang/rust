use super::BackendTypes;
use rustc_middle::mir::coverage::*;
use rustc_middle::ty::Instance;

pub trait CoverageInfoMethods: BackendTypes {
    fn coverageinfo_finalize(&self);
}

pub trait CoverageInfoBuilderMethods<'tcx>: BackendTypes {
    fn create_pgo_func_name_var(&self, instance: Instance<'tcx>) -> Self::Value;

    fn add_counter_region(
        &mut self,
        instance: Instance<'tcx>,
        function_source_hash: u64,
        id: CounterValueReference,
        region: CodeRegion,
    );

    fn add_counter_expression_region(
        &mut self,
        instance: Instance<'tcx>,
        id: InjectedExpressionIndex,
        lhs: ExpressionOperandId,
        op: Op,
        rhs: ExpressionOperandId,
        region: CodeRegion,
    );

    fn add_unreachable_region(&mut self, instance: Instance<'tcx>, region: CodeRegion);
}
