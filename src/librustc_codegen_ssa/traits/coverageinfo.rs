use super::BackendTypes;
use crate::coverageinfo::ExprKind;
use rustc_middle::ty::Instance;

pub trait CoverageInfoMethods: BackendTypes {
    fn coverageinfo_finalize(&self);
}

pub trait CoverageInfoBuilderMethods<'tcx>: BackendTypes {
    fn add_counter_region(
        &mut self,
        instance: Instance<'tcx>,
        function_source_hash: u64,
        index: u32,
        start_byte_pos: u32,
        end_byte_pos: u32,
    );

    fn add_counter_expression_region(
        &mut self,
        instance: Instance<'tcx>,
        index: u32,
        lhs: u32,
        op: ExprKind,
        rhs: u32,
        start_byte_pos: u32,
        end_byte_pos: u32,
    );

    fn add_unreachable_region(
        &mut self,
        instance: Instance<'tcx>,
        start_byte_pos: u32,
        end_byte_pos: u32,
    );
}
