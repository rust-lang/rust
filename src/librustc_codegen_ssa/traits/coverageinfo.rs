use super::BackendTypes;
use crate::coverageinfo::{ExprKind, Region};
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
        index: u32,
        region: Region<'tcx>,
    );

    fn add_counter_expression_region(
        &mut self,
        instance: Instance<'tcx>,
        index: u32,
        lhs: u32,
        op: ExprKind,
        rhs: u32,
        region: Region<'tcx>,
    );

    fn add_unreachable_region(&mut self, instance: Instance<'tcx>, region: Region<'tcx>);
}
